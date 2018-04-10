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

int main(int argc, char* argv[])
{
  uint64_t duration;
  assert(sizeof(ComposedNodeId) == sizeof(uint32_t));
  int required_level = MPI_THREAD_SERIALIZED;
  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  if (argc < 3) {
    cerr << "Usage: " << argv[0] << " <input_graph_path> <output_graph_path> <anonymous_prefix>" << endl;
    return -1;
  }

  string graph_path = argv[1];
  string output_graph_path = argv[2];
  string anonymous_prefix = argv[3];

  Driver* driver = new Driver(MPI_COMM_WORLD, new ObjectPool(anonymous_prefix));
  REGION_BEGIN();
  ConfigFile config(graph_path + ".config");
  uint64_t total_num_nodes = config.get_uint64("total_num_nodes");
  uint64_t total_num_edges = config.get_uint64("total_num_edges");
  GArray<uint32_t>* edges = driver->load_array<uint32_t>(graph_path + ".edges", ObjectMode(UNIFORMITY_UNIFORM_OBJECT, WRITABILITY_READ_ONLY));
  GArray<uint64_t>* index = driver->load_array<uint64_t>(graph_path + ".index", ObjectMode(UNIFORMITY_UNIFORM_OBJECT, WRITABILITY_READ_ONLY));
  MPI_Barrier(MPI_COMM_WORLD);
  assert(edges->size() == total_num_edges);
  assert(index->size() == (total_num_nodes+1));
  SharedGraph<uint32_t, uint64_t> graph(rank, nprocs, edges, index);
  assert(graph.total_num_nodes() == total_num_nodes);
  assert(graph.total_num_edges() == total_num_edges);
  LINES;
  REGION_END("Graph Load");
  REGION_BEGIN();
  GArray<pair<uint32_t, uint32_t>>* graph_tuples = driver->to_tuples(graph);
  LINES;
  REGION_END("Transform CSR To Tuples");
  // delete graph; graph=NULL;
  delete edges; edges=NULL;
  delete index; index=NULL;
  REGION_BEGIN();
  RemapNodeIdMapFn remap_fn(rank, nprocs);
  GArray<pair<uint32_t, uint32_t>>* graph_tuples_remapped = driver->map(graph_tuples, remap_fn);
  LINES;
  REGION_END("Remap ID For Tuples");
  delete graph_tuples; graph_tuples=NULL;
  REGION_BEGIN();
  GArray<pair<uint32_t, uint32_t>>* graph_tuples_reparted = driver->repartition(graph_tuples_remapped, [](pair<uint32_t, uint32_t> edge) {
    uint32_t x = edge.first;
    int part_id = ComposedNodeId(x).partition_id();
    return part_id;
  });
  LINES;
  REGION_END("Repart Tuples");
  delete graph_tuples_remapped; graph_tuples_remapped=NULL;

  REGION_BEGIN();
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
  LINES;
  REGION_END("Local Sort The Tuples");
  REGION_BEGIN();
  auto csr_result = driver->make_csr_from_tuples<uint32_t, uint64_t>(graph_tuples_reparted, [](uint32_t vid) {
    int part_id = ComposedNodeId(vid).partition_id();
    return part_id;
  }, [](uint32_t vid) {
    uint32_t offset = ComposedNodeId(vid).partition_offset();
    return offset;
  }, output_graph_path);
  LINES;
  REGION_END("From Tuple To CSR");
  delete graph_tuples_reparted; graph_tuples_reparted=NULL;
  delete get<0>(csr_result);
  delete get<1>(csr_result);

  MPI_Finalize();
  return 0;
}