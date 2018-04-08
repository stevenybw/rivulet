#ifndef RIVULET_GRAPH_H
#define RIVULET_GRAPH_H

#include <mutex>
#include <thread>

#include <signal.h>

#include "common.h"
#include "util.h"
#include "garray.h"

// #define MPI_DEBUG

#ifdef MPI_DEBUG
#define assert(COND) do{if(!(COND)) {printf("ASSERTION VIOLATED, PROCESS pid = %d PAUSED\n", getpid()); while(1);}}while(0)
static void MPI_Comm_err_handler_function(MPI_Comm* comm, int* errcode, ...) {
  assert(0);
}
#define LINES do{printf("  %d> %s:%d\n", g_rank, __FUNCTION__, __LINE__);}while(0)
static void signal_handler(int sig) {
  printf("SIGNAL %d ENCOUNTERED, PROCESS pid = %d PAUSED\n", sig, getpid());
  while(true);
}
void init_debug() {
  MPI_Errhandler errhandler;
  MPI_Comm_create_errhandler(&MPI_Comm_err_handler_function,  &errhandler);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, errhandler);

  struct sigaction act;
  memset(&act, 0, sizeof(struct sigaction));
  act.sa_handler = signal_handler;
  sigaction(9, &act, NULL);
  sigaction(11, &act, NULL);
}
#else
void init_debug() {}
#endif

inline uint64_t currentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + 1000000 * tv.tv_sec;
}

const double alpha = 0.15;

// const uint64_t g_chunk_size = 64;

int g_group_size = 4;
int g_num_sockets = 4;
int g_rank = 0;
int g_nprocs = 1;

// define converge criterion
// const double epsilon = 0.001;

#define RUN_MODE_PUSH 0
#define RUN_MODE_PULL 1
#define RUN_MODE_DELEGATION_PWQ 2

// enum GraphType { GRAPH_TYPE_SHARED=1, GRAPH_TYPE_DISTRIBUTED=2 };

// typically, NodeT=uint32_t, IndexT=uint64_t
template <typename NodeT, typename IndexT>
struct SharedGraph {
  // vid (global) -> lid (node) -> llid (rank)
  int      _rank;
  int      _nprocs;
  uint64_t _total_num_edges;
  uint64_t _total_num_nodes;
  GArray<NodeT>*  _garr_edges;
  GArray<IndexT>* _garr_index;
  NodeT*   _edges;
  IndexT*  _index;

  // partition info
  uint64_t _vid_chunk_size;
  uint64_t _begin_vid;
  uint64_t _end_vid;

  static void _equally_partition(uint64_t num_elements, int num_parts, int part_id, uint64_t* out_chunk_size, uint64_t* begin_offset, uint64_t* end_offset) {
    uint64_t chunk_size = num_elements / num_parts + num_parts;
    *begin_offset = part_id * chunk_size;
    if (part_id == num_parts - 1) {
      *end_offset = num_elements;
    } else {
      *end_offset = (part_id+1) * chunk_size;
    }
    *out_chunk_size = chunk_size;
  }

  SharedGraph(int rank, int nprocs, GArray<NodeT>* edges, GArray<IndexT>* index) {
    assert(edges->is_uniform() == true);
    assert(index->is_uniform() == true);
    _rank = rank;
    _nprocs = nprocs;
    _total_num_edges = edges->size();
    _total_num_nodes = index->size() - 1;
    _garr_edges = edges;
    _garr_index = index;
    _edges = edges->data();
    _index = index->data();

    _equally_partition(_total_num_nodes, _nprocs, _rank, &_vid_chunk_size, &_begin_vid, &_end_vid);
  }

  int rank() const { return _rank; }
  int nprocs() const { return _nprocs; }

  uint64_t total_num_nodes() const {
    return _total_num_nodes;
  }

  uint64_t total_num_edges() const {
    return _total_num_edges;
  }

  uint64_t local_num_nodes() const {
    return _end_vid - _begin_vid;
  }

  uint64_t local_num_edges() const {
    return _index[_end_vid] - _index[_begin_vid];
  }

  uint64_t get_begin_vid() const {
    return _begin_vid;
  }

  uint64_t get_end_vid() const {
    return _end_vid;
  }

  IndexT get_index_from_vid(uint64_t vid) const {
    return _index[vid];
  }

  IndexT get_index_from_lid(uint64_t lid) const { 
    assert(_begin_vid + lid <= _end_vid);
    return _index[_begin_vid + lid]; 
  }

  NodeT get_edge_from_index(IndexT index) const {
    return _edges[index];
  }

  int get_rank_from_vid(uint64_t vid) const {
    uint64_t pid = vid / _vid_chunk_size;
    assert(pid < (uint64_t) _nprocs);
    return pid;
  }

  uint64_t get_lid_from_vid(uint64_t vid) const {
    uint64_t lid = vid % _vid_chunk_size;
    return lid;
  }

  // GraphType get_type() override { return GRAPH_TYPE_SHARED; }

  static uint64_t _config_read_uint64(ifstream& fin, string name) {
    string token1, token2;
    uint64_t value;
    fin >> token1 >> token2 >> value;
    assert(token1 == name);
    assert(token2 == "=");
    return value;
  }
  
//   void load_csr(const string& prefix, bool in_memory=false) {
//     int rank, nprocs;
//     MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//     MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
// 
//     uint64_t conf_num_parts;
//     uint64_t conf_total_num_nodes;
//     uint64_t conf_total_num_edges;
//     MPI_Barrier(MPI_COMM_WORLD);
//     if(rank==0) {cout << "Loading " << prefix << endl;}
//     {
//       string config_path = prefix + ".config";
//       ifstream fin(config_path);
//       if (!fin) {
//         cerr << "Failed to open config file: " << config_path << endl;
//         assert(false);
//       }
//       conf_num_parts = _config_read_uint64(fin, "num_parts");
//       conf_total_num_nodes = _config_read_uint64(fin, "total_num_nodes");
//       conf_total_num_edges = _config_read_uint64(fin, "total_num_edges");
//     }
//     _rank      = rank;
//     _nprocs    = nprocs;
//     _part_id   = rank / g_group_size;
//     _num_parts = nprocs / g_group_size;
//     assert((uint64_t) _num_parts == conf_num_parts);
//     _slice_id  = rank % g_group_size;
//     _num_slices = g_group_size;
//     _equally_partition(conf_total_num_nodes, _num_parts, _part_id, &_vid_chunk_size, &_begin_vid, &_end_vid);
//     _equally_partition(_end_vid - _begin_vid, _num_slices, _slice_id, &_lid_chunk_size, &_begin_lid, &_end_lid);
// 
//     {
//       string edges_path = prefix + ".edges";
//       MappedFile edges_file;
//       edges_file.open(edges_path.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_SEQUENTIAL);
//       assert(edges_file._bytes % sizeof(NodeT) == 0);
//       _num_edges = edges_file._bytes / sizeof(NodeT);
//       _edges = (NodeT*) edges_file._addr;
// 
//       if (in_memory) {
//         size_t bytes = edges_file._bytes;
//         MappedMemory edges_mem;
//         edges_mem.alloc(bytes, ACCESS_PATTERN_NORMAL);
//         memcpy(edges_mem._addr, edges_file._addr, bytes);
//         _edges = (NodeT*) edges_mem._addr;
//         edges_file.close();
//       }
//     }
//     {
//       string index_path = prefix + ".index";
//       MappedFile index_file;
//       index_file.open(index_path.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_SEQUENTIAL);
//       assert(index_file._bytes % sizeof(IndexT) == 0);
//       _num_nodes = index_file._bytes / sizeof(IndexT) - 1;
//       _index = (IndexT*) index_file._addr;
// 
//       if (in_memory) {
//         size_t bytes = index_file._bytes;
//         MappedMemory index_mem;
//         index_mem.alloc(bytes, ACCESS_PATTERN_NORMAL);
//         memcpy(index_mem._addr, index_file._addr, bytes);
//         _index = (IndexT*) index_mem._addr;
//         index_file.close();
//       }
//     }
//     assert(_num_nodes == (_end_vid - _begin_vid));
//     MPI_Allreduce(&_num_edges, &_total_num_edges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
//     _total_num_edges /= g_group_size;
//     assert(_total_num_edges == conf_total_num_edges);
//     MPI_Allreduce(&_num_nodes, &_total_num_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
//     _total_num_nodes /= g_group_size;
//     assert(_total_num_nodes == conf_total_num_nodes);
// 
//     if (rank == 0) {
//       printf("  num_parts = %d\n", _num_parts);
//       printf("  total_num_nodes = %llu\n", _total_num_nodes);
//       printf("  total_num_edges = %llu\n", _total_num_edges);
//     }
//   }
};

// NodeT:  type for vertex id
// IndexT: type for index
// partition_id_bits: number of partition id bits in real representation: (partition_id_bits : partition_offset_bits)
template <typename NodeT, typename IndexT, size_t partition_id_bits>
struct DistributedGraph
{
  const static size_t partition_offset_bits = 8 * sizeof(NodeT) - partition_id_bits;
  const static size_t partition_offset_mask = (1LL << partition_offset_bits) - 1;
  const static size_t max_num_partitions = (1LL << partition_id_bits);

  int         _rank;
  int         _nprocs;
  uint64_t    _num_edges;
  uint64_t    _num_nodes;
  GArray<NodeT>* _garr_edges;
  GArray<IndexT>* _garr_index;
  NodeT*      _edges;
  IndexT*     _index;
  uint64_t    _total_num_edges;
  uint64_t    _total_num_nodes;

  // GraphType get_type() override { return GRAPH_TYPE_DISTRIBUTED; }

  //GArray<pair<NodeT, NodeT>>* to_tuples(ObjectPoolContext& obj_ctx, string obj_name = "") override {
  //  assert(false /* not implemented */); 
  //}

  DistributedGraph(int rank, int nprocs, GArray<NodeT>* edges, GArray<IndexT>* index) {
    assert(edges->is_separated() == true);
    assert(index->is_separated() == true);
    // _comm = obj_ctx.get_comm();
    _rank = rank;
    _nprocs = nprocs;
    _num_edges = edges->size();
    _num_nodes = index->size()-1;
    _garr_edges = edges;
    _garr_index = index;
    _edges = edges->data();
    _index = index->data();
  }
};


// InputNodeIdT: Exposed type for node id
// IndexT: Exposed type for 
// partition_id_bits: number of partition id bits in real representation
// template <typename InputNodeIdT, typename RealNodeIdT, typename IndexT, size_t partition_id_bits, size_t chunk_size>
// struct DistributedGraph
// {
//   enum {PHASE_READ_ONLY, PHASE_MODIFY_ONLY};
//   // number of partition offset bits in real representation
//   const static size_t partition_offset_bits = 8 * sizeof(RealNodeIdT) - partition_id_bits;
//   const static size_t partition_offset_mask = (1LL << partition_offset_bits) - 1;
// 
//   // max number of paticipating partitions
//   const static size_t max_num_partitions = (1LL << partition_id_bits);
// 
//   struct NodeId {
//     RealNodeIdT data;
// 
//     NodeId (uint32_t pid, uint32_t poffset) : data((pid << partition_offset_bits) + poffset) { }
// 
//     uint32_t partition_id() { return data >> partition_offset_bits; }
//     uint32_t partition_offset() { return data & partition_offset_mask; }
//   };
// 
//   int      state;
//   std::mutex mu;
//   MPI_Comm comm;
//   int      rank;
//   int      nprocs;
//   string   namespace_prefix;
//   uint64_t total_num_nodes;
//   uint64_t total_num_edges;
//   uint64_t local_num_nodes;
//   uint64_t local_num_edges;
//   IndexT*  index_list;
//   NodeId*  node_list;
// 
//   DistributedGraph(); // TODO
// 
//   static NodeId address_resolve(InputNodeIdT vid, int nprocs) {
//     int pid = (vid / chunk_size) % nprocs;
//     uint32_t cid = (vid / chunk_size) / nprocs;
//     uint32_t coff = vid % chunk_size;
//     uint32_t poffset = cid * chunk_size + coff;
//     return NodeId(pid, poffset);
//   }
// 
//   void check() {
//     assert(nprocs <= max_num_partitions);
//     assert(partition_id_bits < 8 * sizeof(RealNodeIdT));
//     assert(sizeof(NodeId) == sizeof(RealNodeIdT));
//   }
// 
//   void begin() {
//     state = PHASE_READ_ONLY;
// 
//   }
// 
//   // create a new vertex and return its id.
//   void add_edge(GraphContext& ctx, InputNodeIdT src, InputNodeIdT dst) {
//     NodeId x(src);
//     NodeId y(dst);
//     ctx.issue_add_edge(x, y);
//   }
// 
//   // collective operation: exchange & update
//   void commit(GraphContext& ctx) {
//     ctx.commit();
//   }
// };

#endif