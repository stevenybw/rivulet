#include <iostream>

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "driver.h"
#include "graph_context.h"

using namespace std;

const size_t partition_id_bits = 1;

const size_t remap_chunk_size_po2 = 6; // 64
const size_t remap_chunk_size     = (1LL << remap_chunk_size_po2);

// number of partition offset bits in real representation
const size_t partition_offset_bits = 8 * sizeof(uint32_t) - partition_id_bits;
const size_t partition_offset_mask = (1LL << partition_offset_bits) - 1;

// max number of paticipating partitions
const size_t max_num_partitions = (1LL << partition_id_bits);

struct ComposedNodeId {
  uint32_t data;

  ComposedNodeId (uint32_t vid) : data(vid) {}
  ComposedNodeId (uint32_t pid, uint32_t poffset) : data((pid << partition_offset_bits) + poffset) {}

  uint32_t partition_id() { return data >> partition_offset_bits; }
  uint32_t partition_offset() { return data & partition_offset_mask; }
};

struct RemapNodeIdMapFn : public MapFn<pair<uint32_t, uint32_t>, pair<uint32_t, uint32_t>>
{
  int rank, nprocs;

  RemapNodeIdMapFn(int rank, int nprocs) : rank(rank), nprocs(nprocs) {}

  uint32_t map_vertex(uint32_t vid) {
    int      pid = (vid / remap_chunk_size) % nprocs; // process id
    uint32_t cid = (vid / remap_chunk_size) / nprocs; // chunk id inside a process
    uint32_t coff = vid % remap_chunk_size;           // offset in the chunk
    uint32_t poffset = cid * remap_chunk_size + coff; // offset in the process
    uint32_t new_node_id = ComposedNodeId(pid, poffset).data;
    return new_node_id;
  }

  OutputType processElement(const InputType& in_element) override {
    return make_pair(map_vertex(in_element.first), map_vertex(in_element.second));
  }
};

struct EqualVertexRemapNodeIdMapFn : public MapFn<pair<uint32_t, uint32_t>, pair<uint32_t, uint32_t>>
{
  int rank, nprocs;
  uint64_t total_num_vertices;
  uint64_t vertex_chunk;

  EqualVertexRemapNodeIdMapFn(int rank, int nprocs, uint64_t total_num_vertices) : rank(rank), nprocs(nprocs), total_num_vertices(total_num_vertices), vertex_chunk(total_num_vertices / nprocs) {}

  uint32_t map_vertex(uint32_t vid) {
    int pid = vid / vertex_chunk;
    uint32_t poffset = vid % vertex_chunk;
    uint32_t new_node_id = ComposedNodeId(pid, poffset).data;
    return new_node_id;
  }

  OutputType processElement(const InputType& in_element) override {
    return make_pair(map_vertex(in_element.first), map_vertex(in_element.second));
  }
};

const char* METHOD_EQUAL_VERTEX = "equal_vertex"; 
const char* METHOD_EQUAL_VERTEX = "equal_edges"; 
const char* METHOD_CHUNK_ROUNDROBIN = "chunkroundrobin";

int main(int argc, char* argv[])
{
  uint64_t duration;
  assert(sizeof(ComposedNodeId) == sizeof(uint32_t));
  int required_level = MPI_THREAD_SERIALIZED;
  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  assert(provided_level >= required_level);
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  if (argc < 4) {
    cerr << "Usage: " << argv[0] << " <input_graph_path in NFS> <output_graph_path> <method>" << endl;
    cerr << "  method: {equal_vertex, equal_edges, chunkroundrobin}" << endl;
    return -1;
  }

  string graph_path = argv[1];
  string output_graph_path = argv[2];
  string method = argv[3];

  Configuration env_config;
  ExecutionContext ctx(env_config.nvm_off_cache_pool_dir, env_config.nvm_off_cache_pool_dir, env_config.nvm_on_cahce_pool_dir, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);
  LOG_BEGIN();
  LOG_INFO("Loading the graph");
  ConfigFile config(graph_path + ".config");
  uint64_t total_num_nodes = config.get_uint64("total_num_nodes");
  uint64_t total_num_edges = config.get_uint64("total_num_edges");
  GArray<pair<uint32_t, uint32_t>>* graph_tuples = driver->readFromBinaryRecords<pair<uint32_t, uint32_t>>(graph_path + ".tuples");
  if (rank == 0) {
    printf("Remap ID For Tuples Using %s\n", method.c_str());
  }
  GArray<pair<uint32_t, uint32_t>>* graph_tuples_remapped = NULL;
  if (method == METHOD_CHUNK_ROUNDROBIN) {
    RemapNodeIdMapFn remap_fn(rank, nprocs);
    graph_tuples_remapped = driver->map(graph_tuples, remap_fn, ObjectRequirement::create_transient(Object::NVM_ON_CACHE)); // MPI does not support DAX...
  } else if (method == METHOD_EQUAL_VERTEX) {
    EqualVertexRemapNodeIdMapFn remap_fn(rank, nprocs, total_num_nodes);
    graph_tuples_remapped = driver->map(graph_tuples, remap_fn, ObjectRequirement::create_transient(Object::NVM_ON_CACHE)); // MPI does not support DAX...
  } else {
    assert(false);
  }
  LINES;
  delete graph_tuples; graph_tuples=NULL;

  LOG_INFO("Repart Tuples");
  GArray<pair<uint32_t, uint32_t>>* graph_tuples_reparted = driver->repartition(graph_tuples_remapped, [](pair<uint32_t, uint32_t> edge) {
    uint32_t x = edge.first;
    int part_id = ComposedNodeId(x).partition_id();
    return part_id;
  }, ObjectRequirement::create_transient(Object::NVM_ON_CACHE));
  delete graph_tuples_remapped; graph_tuples_remapped=NULL;
  LOG_INFO("Local Sort The Tuples");
  {
    pair<uint32_t, uint32_t>* graph_tuples_reparted_arr = graph_tuples_reparted->data();
    size_t graph_tuples_reparted_ne = graph_tuples_reparted->size();
    my_sort(graph_tuples_reparted_arr, graph_tuples_reparted_arr+graph_tuples_reparted_ne, 
      [](const pair<uint32_t, uint32_t>& lhs, const pair<uint32_t, uint32_t>& rhs) {
        if (lhs.first < rhs.first) {
          return true;
        } else if (lhs.first == rhs.first && lhs.second < rhs.second) {
          return true;
        }
        return false;
    });
  }
  LOG_INFO("From Tuple To CSR");
  auto csr_result = driver->make_csr_from_tuples<uint32_t, uint64_t>(graph_tuples_reparted, 
    [](uint32_t vid) {
      int part_id = ComposedNodeId(vid).partition_id();
      return part_id;
    }, 
    [](uint32_t vid) {
      uint32_t offset = ComposedNodeId(vid).partition_offset();
      return offset;
    },
    ObjectRequirement::create_persist(output_graph_path+".edges"),
    ObjectRequirement::create_persist(output_graph_path+".index"));
  
  delete graph_tuples_reparted; graph_tuples_reparted=NULL;
  delete get<0>(csr_result);
  delete get<1>(csr_result);

  MPI_Finalize();
  return 0;
}