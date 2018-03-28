// Simpler PageRank: no fault tolerant, single machine.

#include <fstream>
#include <iostream>
#include <iomanip>
#include <thread>

#include <assert.h>
#include <fcntl.h>
#include <immintrin.h>
#include <mpi.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <signal.h>

#include <numa.h>
#include <numaif.h>

using namespace std;

#define MPI_DEBUG

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

#include "channel.h"
#include "file.h"
#include "util.h"

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

// typically, NodeT=uint32_t, IndexT=uint64_t
template <typename NodeT, typename IndexT>
struct Graph {
  // vid (global) -> lid (node) -> llid (rank)
  int      _rank;
  int      _nprocs;
  int      _part_id;
  int      _num_parts;
  int      _slice_id;
  int      _num_slices;

  // level 1
  uint64_t _vid_chunk_size;
  uint64_t _begin_vid;
  uint64_t _end_vid;

  // level 2
  uint64_t _lid_chunk_size;
  uint64_t _begin_lid;
  uint64_t _end_lid;

  uint64_t _total_num_edges;
  uint64_t _total_num_nodes;

  // graph partition
  uint64_t _num_nodes;
  IndexT*  _index;
  uint64_t _num_edges;
  NodeT*   _edges;

  uint64_t num_local_nodes() const { return _end_lid - _begin_lid; }
  IndexT get_index_from_llid(uint64_t llid) const { 
    assert(_begin_lid + llid <= _num_nodes);
    return _index[_begin_lid + llid]; 
  }
  NodeT get_edge_from_index(IndexT index) const {
    return _edges[index];
  }
  int get_rank_from_vid(uint64_t vid) const {
    uint64_t pid = vid / _vid_chunk_size;
    assert(pid < (uint64_t) _num_parts);
    uint64_t lid = vid % _vid_chunk_size;
    uint64_t rank = pid * g_group_size + lid / _lid_chunk_size;
    assert(rank < (uint64_t) g_nprocs);
    return rank;
  }
  uint64_t get_llid_from_vid(uint64_t vid) const {
    uint64_t lid = vid % _vid_chunk_size;
    uint64_t llid = lid % _lid_chunk_size;
    return llid;
  }

  static uint64_t _config_read_uint64(ifstream& fin, string name) {
    string token1, token2;
    uint64_t value;
    fin >> token1 >> token2 >> value;
    assert(token1 == name);
    assert(token2 == "=");
    return value;
  }

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
  
  void load_csr(const string& prefix, bool in_memory=false) {
    int rank, nprocs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    uint64_t conf_num_parts;
    uint64_t conf_total_num_nodes;
    uint64_t conf_total_num_edges;
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) {cout << "Loading " << prefix << endl;}
    {
      string config_path = prefix + ".config";
      ifstream fin(config_path);
      if (!fin) {
        cerr << "Failed to open config file: " << config_path << endl;
        assert(false);
      }
      conf_num_parts = _config_read_uint64(fin, "num_parts");
      conf_total_num_nodes = _config_read_uint64(fin, "total_num_nodes");
      conf_total_num_edges = _config_read_uint64(fin, "total_num_edges");
    }
    _rank      = rank;
    _nprocs    = nprocs;
    _part_id   = rank / g_group_size;
    _num_parts = nprocs / g_group_size;
    assert((uint64_t) _num_parts == conf_num_parts);
    _slice_id  = rank % g_group_size;
    _num_slices = g_group_size;
    _equally_partition(conf_total_num_nodes, _num_parts, _part_id, &_vid_chunk_size, &_begin_vid, &_end_vid);
    _equally_partition(_end_vid - _begin_vid, _num_slices, _slice_id, &_lid_chunk_size, &_begin_lid, &_end_lid);

    {
      string edges_path = prefix + ".edges";
      MappedFile edges_file;
      edges_file.open(edges_path.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_SEQUENTIAL);
      assert(edges_file._bytes % sizeof(NodeT) == 0);
      _num_edges = edges_file._bytes / sizeof(NodeT);
      _edges = (NodeT*) edges_file._addr;

      if (in_memory) {
        size_t bytes = edges_file._bytes;
        MappedMemory edges_mem;
        edges_mem.alloc(bytes, ACCESS_PATTERN_NORMAL);
        memcpy(edges_mem._addr, edges_file._addr, bytes);
        _edges = (NodeT*) edges_mem._addr;
        edges_file.close();
      }
    }
    {
      string index_path = prefix + ".index";
      MappedFile index_file;
      index_file.open(index_path.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_SEQUENTIAL);
      assert(index_file._bytes % sizeof(IndexT) == 0);
      _num_nodes = index_file._bytes / sizeof(IndexT) - 1;
      _index = (IndexT*) index_file._addr;

      if (in_memory) {
        size_t bytes = index_file._bytes;
        MappedMemory index_mem;
        index_mem.alloc(bytes, ACCESS_PATTERN_NORMAL);
        memcpy(index_mem._addr, index_file._addr, bytes);
        _index = (IndexT*) index_mem._addr;
        index_file.close();
      }
    }
    assert(_num_nodes == (_end_vid - _begin_vid));
    MPI_Allreduce(&_num_edges, &_total_num_edges, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    _total_num_edges /= g_group_size;
    assert(_total_num_edges == conf_total_num_edges);
    MPI_Allreduce(&_num_nodes, &_total_num_nodes, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    _total_num_nodes /= g_group_size;
    assert(_total_num_nodes == conf_total_num_nodes);
    
    if (rank == 0) {
      printf("  num_parts = %d\n", _num_parts);
      printf("  total_num_nodes = %llu\n", _total_num_nodes);
      printf("  total_num_edges = %llu\n", _total_num_edges);
    }
  }
};

struct AccumulateRequest {
  double* target;
  double rhs;
};

static constexpr int BATCH_SIZE = 32;
static thread_local size_t tl_num_reqs = 0;
static thread_local AccumulateRequest tl_acc_reqs[BATCH_SIZE];

struct PageRankAggregator {
  const static double zero_value;
  
  static void update (double* p_lhs, double rhs) {
    (*p_lhs) += rhs;
  }

  static void cas_atomic_update(double* p_lhs, double rhs) {
    double lhs = *p_lhs;
    // uint64_t i_lhs = *((uint64_t*) &lhs);
    // dereferencing type-punned pointer will break strict-aliasing rules
    uint64_t i_lhs;
    memcpy(&i_lhs, &lhs, sizeof(uint64_t));
    while(true) {
      // total_attempt++;
      double new_lhs = lhs + rhs;
      uint64_t i_new_lhs;
      memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
      uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
      if (i_current_lhs == i_lhs) {
        break;
      }
      i_lhs = i_current_lhs;
      memcpy(&lhs, &i_lhs, sizeof(uint64_t));
      // num_retry++;
    }
  }

  static void rtm_atomic_update(double* p_lhs, double rhs) {
    // total_attempt++;
    if(_xbegin() == (unsigned int) -1) {
      *p_lhs += rhs;
      _xend();
    } else {
      _mm_pause();
      double lhs = *p_lhs;
      uint64_t i_lhs;
      memcpy(&i_lhs, &lhs, sizeof(uint64_t));
      while(true) {
        // total_attempt++;
        double new_lhs = lhs + rhs;
        uint64_t i_new_lhs;
        memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
        uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
        if (i_current_lhs == i_lhs) {
          break;
        }
        i_lhs = i_current_lhs;
        memcpy(&lhs, &i_lhs, sizeof(uint64_t));
        // num_retry++;
        _mm_pause();
      }
    }
  }

  static void batched_atomic_update_flush(const int num_reqs) {
    // total_attempt++;
    if(_xbegin() == (unsigned int) -1) {
      for(int i=0; i<num_reqs; i++) {
        *(tl_acc_reqs[i].target) += tl_acc_reqs[i].rhs;
      }
      _xend();
    } else {
      _mm_pause();
      for(int i=0; i<num_reqs; i++) {
        double* p_lhs = tl_acc_reqs[i].target;
        double rhs = tl_acc_reqs[i].rhs;

        double lhs = *p_lhs;
        uint64_t i_lhs;
        memcpy(&i_lhs, &lhs, sizeof(uint64_t));
        while(true) {
          // total_attempt++;
          double new_lhs = lhs + rhs;
          uint64_t i_new_lhs;
          memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
          uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
          if (i_current_lhs == i_lhs) {
            break;
          }
          i_lhs = i_current_lhs;
          memcpy(&lhs, &i_lhs, sizeof(uint64_t));
          // num_retry++;
          _mm_pause();
        }
      }
    }
    tl_num_reqs = 0;
  }

  static void batched_atomic_update(double* p_lhs, double rhs) {
    tl_acc_reqs[tl_num_reqs].target = p_lhs;
    tl_acc_reqs[tl_num_reqs].rhs = rhs;
    tl_num_reqs++;
    if (tl_num_reqs == BATCH_SIZE) {
      batched_atomic_update_flush(BATCH_SIZE);
    }
  }
};

void pin_memory(void* ptr, size_t size) {
  if (mlock(ptr, size) < 0) {
    perror("mlock failed");
    assert(false);
  }
}

void* pages[128*1024];
int nodes[128*1024];
int statuses[128*1024];

void interleave_memory(void* ptr, size_t size, size_t chunk_size, int* node_list, int num_nodes) {
  printf("interleave begin");
  assert(chunk_size == 4096);
  char* buf = (char*) ptr;
  size_t count = 0;
  for(size_t pos=0; pos<size; pos+=chunk_size) {
    pages[count] = &buf[pos];
    nodes[count] = node_list[count % num_nodes];
    statuses[count] = -1;
    count++;
  }
  int ok = move_pages(0, count, pages, nodes, statuses, MPOL_MF_MOVE);
  assert(ok != -1);
  for(size_t i=0; i<count; i++) {
    assert(statuses[i] == node_list[i % num_nodes]);
  }
  printf("interleave end");
}

void* malloc_pinned(size_t size) {
  // !NOTE: use memalign to allocate 256-bytes aligned buffer
  void* ptr = am_memalign(4096, size);
  if (mlock(ptr, size)<0) {
    perror("mlock failed");
    assert(false);
  }
  return ptr;
}

template <typename NodeT, typename IndexT>
void RunPageRankPush(Graph<NodeT, IndexT>& graph, double* src, double* next_val, uint32_t chunk_size) {
  size_t num_nodes = graph._num_nodes;
  #pragma omp parallel
  {
    uint32_t num_parts = num_nodes/chunk_size;
    #pragma omp for schedule(dynamic, 1)
    for (uint32_t part_id=0; part_id<num_parts; part_id++) {
      uint32_t chunk_begin = part_id*chunk_size;
      uint32_t chunk_end;
      if (part_id == num_parts - 1) {
        chunk_end = num_nodes;
      } else {
        chunk_end = (part_id+1)*chunk_size;
      }
      for(uint32_t i=chunk_begin; i<chunk_end; i++) {
        uint64_t from = graph._index[i];
        uint64_t to   = graph._index[i+1];
        double contrib = src[i];
        for (uint64_t idx=from; idx<to; idx++) {
          uint32_t y = graph._edges[idx];
          #if defined(UPDATE_SEQUENTIAL)
          PageRankAggregator::update(&next_val[y], contrib);
          #elif defined(UPDATE_CAS)
          PageRankAggregator::cas_atomic_update(&next_val[y], contrib);
          #elif defined(UPDATE_BATCHED_RTM)
          PageRankAggregator::batched_atomic_update(&next_val[y], contrib);
          #else
          #error "Specify a mode"
          #endif
        }
      }
    }
    #if defined(UPDATE_BATCHED_RTM)
    PageRankAggregator::batched_atomic_update_flush(tl_num_reqs);
    #endif
  }
}

const size_t channel_num = 2;
//const size_t channel_bytes = 4096;
const size_t channel_bytes = 4096;
// const size_t channel_bytes = 32768;
const size_t updater_block_size_po2 = 4;

const size_t MPI_RECV_BUFFER_SIZE = 1*1024*1024;
const size_t MPI_SEND_BUFFER_SIZE = 1*1024*1024;
const int MAX_CLIENT_THREADS = 16;
const int MAX_UPDATER_THREADS = 16;
const int MAX_IMPORT_THREADS = 4;
const int MAX_EXPORT_THREADS = 4;
const int TAG_DATA = 100;
const int TAG_CLOSE = 101;

//using DefaultChannel = Channel<2, channel_bytes>;
using DefaultChannel = Channel_1<channel_bytes>;

bool is_power_of_2(uint64_t num) {
  while (num>0 && (num&1)==0) {
    num >>= 1;
  }
  if (num == 1) {
    return true;
  } else {
    return false;
  }
}

struct __attribute__((packed)) UpdateRequest {
  uint32_t y;
  double contrib;
};

DefaultApproxMod approx_mod; // for updater
DefaultChannel channels[MAX_CLIENT_THREADS][MAX_UPDATER_THREADS];
DefaultChannel to_exports[MAX_CLIENT_THREADS][MAX_EXPORT_THREADS];
DefaultChannel from_imports[MAX_IMPORT_THREADS][MAX_UPDATER_THREADS];
volatile int* num_client_done; // number of local client done
volatile int* num_import_done; // number of local importer done
volatile int* current_chunk_id; // shared chunk_id for chunked work stealing
volatile int* importer_num_close_request; // shared among importers, number of close requests received

void AM_Init(LaunchConfig config) {
  int num_sockets = config.num_sockets;
  int num_updater_sockets = config.num_updater_sockets;
  int num_comm_sockets    = config.num_comm_sockets;
  int num_client_threads_per_socket = config.num_client_threads_per_socket;
  int num_updater_threads_per_socket = config.num_updater_threads_per_socket;
  int num_import_threads_per_socket = config.num_import_threads_per_socket;
  int num_export_threads_per_socket = config.num_export_threads_per_socket;
  int num_client_threads = num_client_threads_per_socket * num_sockets;
  int num_updater_threads = num_updater_threads_per_socket * num_updater_sockets;
  int num_import_threads  = num_import_threads_per_socket * num_comm_sockets;
  int num_export_threads  = num_export_threads_per_socket * num_comm_sockets;

  assert(num_client_threads  <= MAX_CLIENT_THREADS);
  assert(num_updater_threads <= MAX_UPDATER_THREADS);
  assert(num_import_threads  <= MAX_IMPORT_THREADS);
  assert(num_export_threads  <= MAX_EXPORT_THREADS);
  assert(is_power_of_2(num_import_threads));
  assert(is_power_of_2(num_export_threads));

  approx_mod.init(num_updater_threads);
  for(int i=0; i<num_client_threads; i++) {
    for(int j=0; j<num_updater_threads; j++) {
      channels[i][j].init();
    }
  }
  for(int i=0; i<num_client_threads; i++) {
    for(int j=0; j<num_export_threads; j++) {
      to_exports[i][j].init();
    }
  }
  for(int i=0; i<num_import_threads; i++) {
    for(int j=0; j<num_updater_threads; j++) {
      from_imports[i][j].init();
    }
  }
  num_client_done = new int;
  num_import_done = new int;
  current_chunk_id = new int;
  importer_num_close_request = new int;
}

#define ROL(val, bits) (((val)>>(64-(bits)))|((val)<<(bits)))
#define ROR(val, bits) (((val)>>(bits))|((val)<<(64-(bits))))

template <typename T>
struct __attribute__((packed)) GeneralUpdateRequest {
  uint32_t y;
  T contrib;
};

template <typename NodeT, typename IndexT, typename VertexT, typename UpdateCallback>
void edge_map(LaunchConfig config, Graph<NodeT, IndexT>& graph, VertexT* curr_val, VertexT* next_val, uint32_t chunk_size, const UpdateCallback& update_op)
{
  using UpdateRequest = GeneralUpdateRequest<VertexT>;

  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  // size_t   num_nodes          = graph._num_nodes;
  // int      part_id            = graph._part_id;
  // int      num_parts          = graph._num_parts;

  if (rank == 0) {
    cout << "  graph vertex num = " << graph._num_nodes << endl;
    cout << "  graph edges num = " << graph._num_edges << endl;
  }

  *num_client_done = 0;
  *num_import_done = 0;
  *current_chunk_id = 0;
  *importer_num_close_request = 0;
  int num_sockets = config.num_sockets;
  int num_updater_sockets = config.num_updater_sockets;
  int num_comm_sockets = config.num_comm_sockets;
  int num_client_threads_per_socket = config.num_client_threads_per_socket;
  int num_updater_threads_per_socket = config.num_updater_threads_per_socket;
  int num_import_threads_per_socket = config.num_import_threads_per_socket;
  int num_export_threads_per_socket = config.num_export_threads_per_socket;
  int num_client_threads  = num_client_threads_per_socket * num_sockets;
  int num_updater_threads = num_updater_threads_per_socket * num_updater_sockets;
  int num_import_threads  = num_import_threads_per_socket * num_comm_sockets;
  int num_export_threads  = num_export_threads_per_socket * num_comm_sockets;
  int* socket_list = config.socket_list;
  int* updater_socket_list = config.updater_socket_list;
  int* comm_socket_list = config.comm_socket_list;

  if (rank == 0) {
    cout << "  channel_num = " << channel_num << endl;
    cout << "  channel_bytes = " << channel_bytes << endl;
    cout << "  updater_block_size = " << (1LL<<updater_block_size_po2) << endl;
  }

  if (num_client_threads < 1 || num_updater_threads < 1) {
    cout << "Insufficient number of threads: " << num_client_threads << ", " << num_updater_threads << endl;
    assert(false);
  }

  std::thread updater_threads[num_updater_threads];
  for (int i=0; i<num_updater_threads; i++) {
    updater_threads[i] = std::move(std::thread([&update_op](int id, int socket_id, int num_client_threads, int num_import_threads, volatile int* client_dones, volatile int* import_dones, VertexT* next_val) {
      uint64_t num_updates = 0;
      auto process_callback = [next_val, &update_op, &num_updates](const char* ptr, size_t bytes) {
        num_updates += bytes / sizeof(UpdateRequest);
        assert(bytes % sizeof(UpdateRequest) == 0);
        const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
        for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
          uint64_t llid = req_ptr->y;
          VertexT contrib = req_ptr->contrib;
          update_op(&next_val[llid], contrib);
          req_ptr++;
        }
      };
      int ok = numa_run_on_node(socket_id);
      assert(ok == 0);
      while(!(*client_dones == num_client_threads && *import_dones == num_import_threads)) {
        for(int from=0; from<num_client_threads; from++) {
          channels[from][id].poll(process_callback);
        }
        for(int import_id=0; import_id<num_import_threads; import_id++) {
          from_imports[import_id][id].poll(process_callback);
        }
      }
      printf(">>  updater thread %d request = %lu\n", id, num_updates);
    }, i, updater_socket_list[i/num_updater_threads_per_socket], num_client_threads, num_import_threads, num_client_done, num_import_done, next_val));
  }

  std::thread export_threads[num_export_threads];
  for (int i=0; i<num_export_threads; i++) {
    export_threads[i] = std::move(std::thread([&graph](int tid, int socket_id, int nprocs, int num_client_threads, volatile int* client_dones) {
      int ok = numa_run_on_node(socket_id);
      assert(ok == 0);
      int          buf_id_list[nprocs];
      size_t       curr_bytes_list[nprocs];
      MPI_Request  req[nprocs][2];
      char*        sendbuf[nprocs][2];
      for(int p=0; p<nprocs; p++) {
        buf_id_list[p] = 0;
        curr_bytes_list[p] = 0;
        for(int i=0; i<2; i++) {
          req[p][i] = MPI_REQUEST_NULL;
          sendbuf[p][i] = (char*) am_memalign(64, MPI_SEND_BUFFER_SIZE);
          assert(sendbuf[p][i]);
        }
      }
      while(*client_dones != num_client_threads) {
        for(int from=0; from<num_client_threads; from++) {
          to_exports[from][tid].poll([&graph, &buf_id_list, &curr_bytes_list, &req, &sendbuf](const char* ptr, size_t bytes) {
            assert(bytes % sizeof(UpdateRequest) == 0);
            const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
            for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
              uint64_t y = req_ptr->y;
              int next_val_rank = graph.get_rank_from_vid(y);
              uint32_t next_val_llid = graph.get_llid_from_vid(y);
              UpdateRequest update_request;
              update_request.y = next_val_llid;
              update_request.contrib = req_ptr->contrib;
              {
                int    buf_id = buf_id_list[next_val_rank];
                size_t curr_bytes = curr_bytes_list[next_val_rank];
                if (curr_bytes + sizeof(UpdateRequest) > MPI_SEND_BUFFER_SIZE) {
                  // LINES;
                  int flag = 0;
                  // printf("  %d> send %zu bytes to %d\n", g_rank, curr_bytes, next_val_rank);
                  MPI_Isend(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD, &req[next_val_rank][buf_id]);
                  while (!flag) {
                    MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
                  }
                  // MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
                  buf_id = buf_id ^ 1;
                  curr_bytes = 0;
                  curr_bytes_list[next_val_rank] = curr_bytes;
                  buf_id_list[next_val_rank] = buf_id;
                }
                memcpy(&sendbuf[next_val_rank][buf_id][curr_bytes], &update_request, sizeof(UpdateRequest));
                curr_bytes += sizeof(UpdateRequest);
                curr_bytes_list[next_val_rank] = curr_bytes;
              }
              req_ptr++;
            }
          });
        }
      }
      for (int next_val_rank=0; next_val_rank<nprocs; next_val_rank++) {
        int    buf_id = buf_id_list[next_val_rank];
        size_t curr_bytes = curr_bytes_list[next_val_rank];
        int flag = 0;
        while (!flag) {
          MPI_Test(&req[next_val_rank][buf_id], &flag, MPI_STATUS_IGNORE);
        }
        req[next_val_rank][buf_id] = MPI_REQUEST_NULL;
        while (!flag) {
          MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
        }
        req[next_val_rank][buf_id^1] = MPI_REQUEST_NULL;
        MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
        MPI_Send(NULL, 0, MPI_CHAR, next_val_rank, TAG_CLOSE, MPI_COMM_WORLD);
        buf_id = buf_id ^ 1;
        curr_bytes = 0;
        curr_bytes_list[next_val_rank] = curr_bytes;
        buf_id_list[next_val_rank] = buf_id;
      }
    }, i, comm_socket_list[i/num_export_threads_per_socket], nprocs, num_client_threads, num_client_done));
  }

  std::thread import_threads[num_import_threads];
  for (int i=0; i<num_import_threads; i++) {
    import_threads[i] = std::move(std::thread([&graph](int tid, int socket_id, int nprocs, int num_export_threads, int num_updater_threads, volatile int* importer_num_close_request, volatile int* import_dones) {
      int ok = numa_run_on_node(socket_id);
      assert(ok == 0);
      int buf_id = 0;
      MPI_Request req[2];
      char* recvbuf[2];
      short counter[MAX_UPDATER_THREADS];
      memset(counter, 0, sizeof(counter));
      for(int i=0; i<2; i++) {
        recvbuf[i] = (char*) am_memalign(64, MPI_RECV_BUFFER_SIZE);
        assert(recvbuf[i]);
      }
      MPI_Irecv(recvbuf[buf_id], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id]);
      int num_flush = 0;
      while (true) {
        int flag;
        MPI_Status st;
        MPI_Test(&req[buf_id], &flag, &st);
        if (flag) {
          // printf("  %d> RECEIVED FROM %d\n", g_rank, st.MPI_SOURCE);
          MPI_Irecv(recvbuf[buf_id^1], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id^1]);
          // int source = st.MPI_SOURCE;
          int tag    = st.MPI_TAG;
          int rbytes = 0;
          MPI_Get_count(&st, MPI_CHAR, &rbytes);
          if (tag == TAG_DATA) {
            UpdateRequest* rbuf = (UpdateRequest*) recvbuf[buf_id];
            assert(rbytes % sizeof(UpdateRequest) == 0);
            int rcount = rbytes / sizeof(UpdateRequest);
            for(int i=0; i<rcount; i++) {
              uint32_t llid = rbuf[i].y;
              int target_tid = approx_mod.approx_mod(llid >> updater_block_size_po2);
              size_t next_bytes = from_imports[tid][target_tid].push_explicit(rbuf[i], counter[target_tid]);
              counter[target_tid] = next_bytes;
            }
          } else if (tag == TAG_CLOSE) {
            assert(rbytes == 0);
            __sync_fetch_and_add(importer_num_close_request, 1);
            // printf("  %d> CLOSE RECEIVED (%d/%d)\n", g_rank, *importer_num_close_request, num_export_threads * nprocs);
          }
          buf_id = buf_id^1;
          // LINES;
        }
        if (*importer_num_close_request == num_export_threads * nprocs) {
          num_flush++;
          if (num_flush == 2) {
            break;
          }
        }
      }
      // printf("  %d> IMPORT EXIT\n", g_rank);
      MPI_Cancel(&req[buf_id]);
      for (int i=0; i<2; i++) {
        req[i] = MPI_REQUEST_NULL;
        am_free(recvbuf[i]);
        recvbuf[i] = NULL;
      }
      for (int next_val=0; next_val<num_updater_threads; next_val++) {
        size_t curr_bytes = counter[next_val];
        from_imports[tid][next_val].flush_and_wait_explicit(curr_bytes);
        counter[next_val] = 0;
      }
      __sync_fetch_and_add(import_dones, 1);
    }, i, comm_socket_list[i/num_import_threads_per_socket], nprocs, num_export_threads, num_updater_threads, importer_num_close_request, num_import_done));
  }

  uint64_t edges_processed_per_thread[num_client_threads];
  uint64_t num_local_nodes = graph.num_local_nodes();
  uint32_t num_chunks      = num_local_nodes/chunk_size;
  *current_chunk_id        = 0;

  #pragma omp parallel num_threads(num_client_threads)
  {
    //register __m128i thread_local_state asm ("xmm15");
    //thread_local_state[0] = 0;
    short counter[MAX_UPDATER_THREADS];
    short export_counter[MAX_EXPORT_THREADS];
    memset(counter, 0, sizeof(counter));
    memset(export_counter, 0, sizeof(export_counter));

    uint64_t edges_processed;
    edges_processed = 0;
    int tid = omp_get_thread_num();
    int socket_id = socket_list[tid/num_client_threads_per_socket];
    int ok  = numa_run_on_node(socket_id);
    assert(ok == 0);
    while (true) {
      uint32_t chunk_id = __sync_fetch_and_add(current_chunk_id, 1);
      if (chunk_id >= num_chunks) {
        break;
      }
      uint32_t chunk_begin = chunk_id*chunk_size;
      uint32_t chunk_end;
      if (chunk_id == num_chunks - 1) {
        chunk_end = num_local_nodes;
      } else {
        chunk_end = (chunk_id+1)*chunk_size;
      }
      for (uint32_t i=chunk_begin; i<chunk_end; i++) {
        uint64_t from  = graph.get_index_from_llid(i);
        uint64_t to    = graph.get_index_from_llid(i+1);
        VertexT  contrib = curr_val[i];
        edges_processed += (to-from);
        for (uint64_t idx=from; idx<to; idx++) {
          uint32_t vid = graph.get_edge_from_index(idx);
          int target_rank = graph.get_rank_from_vid(vid);
          if (target_rank == rank) {
            uint32_t llid = graph.get_llid_from_vid(vid);
            UpdateRequest req;
            req.y=llid;
            req.contrib=contrib;
            int target_tid = approx_mod.approx_mod(llid >> updater_block_size_po2);
            size_t next_bytes = channels[tid][target_tid].push_explicit(req, counter[target_tid]);
            counter[target_tid] = next_bytes;
          } else {
            UpdateRequest req;
            req.y = vid;
            req.contrib = contrib;
            int export_id = target_rank % num_export_threads;
            size_t next_bytes = to_exports[tid][export_id].push_explicit(req, export_counter[export_id]);
            export_counter[export_id] = next_bytes;
          }
        }
      }
    }
    for (int next_val=0; next_val<num_updater_threads; next_val++) {
      size_t curr_bytes = counter[next_val];
      channels[tid][next_val].flush_and_wait_explicit(curr_bytes);
      counter[next_val] = 0;
    }
    for (int export_id=0; export_id<num_export_threads; export_id++) {
      size_t curr_bytes = export_counter[export_id];
      to_exports[tid][export_id].flush_and_wait_explicit(curr_bytes);
      export_counter[export_id] = 0;
    }
    __sync_fetch_and_add(num_client_done, 1);
    edges_processed_per_thread[tid] = edges_processed;
  }

  for(int i=0; i<num_updater_threads; i++) {
    updater_threads[i].join();
  }
  for(int i=0; i<num_export_threads; i++) {
    export_threads[i].join();
  }
  for(int i=0; i<num_import_threads; i++) {
    import_threads[i].join();
  }

  uint64_t sum_ep = 0;
  uint64_t max_ep = 0;
  for(int i=0; i<num_client_threads; i++) {
    uint64_t ep = edges_processed_per_thread[i];
    printf("  %d> client thread %2d: processed edges = %lu\n", g_rank, i, ep);
    if (ep > max_ep) max_ep = ep;
    sum_ep += ep;
  }
  uint64_t global_sum_ep = 0;
  uint64_t global_max_ep = 0;
  MPI_Allreduce(&sum_ep, &global_sum_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
  MPI_Allreduce(&max_ep, &global_max_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
  // printf("  avg_client_processed_edge = %lf\n", 1.0*sum_ep/num_client_threads);
  // printf("  max_client_processed_edge = %lf\n", 1.0*max_ep);
  if (rank == 0) {
    printf(">>> client load imbalance = %lf\n", (1.0*global_max_ep)/(1.0*global_sum_ep/nprocs/num_client_threads));
  }
}

template <typename NodeT, typename IndexT>
void RunPageRankPull(Graph<NodeT, IndexT>& graph_t, double* src, double* next_val, uint32_t chunk_size) {
  printf("PULL\n");
  size_t num_nodes = graph_t._num_nodes;
  #pragma omp parallel
  {
    uint32_t num_parts = num_nodes/chunk_size;
    #pragma omp for schedule(dynamic, 1)
    for (uint32_t part_id=0; part_id<num_parts; part_id++) {
      uint32_t chunk_begin = part_id * chunk_size;
      uint32_t chunk_end;
      if (part_id == num_parts - 1) {
        chunk_end = num_nodes;
      } else {
        chunk_end = (part_id+1)*chunk_size;
      }
      for(uint32_t y=chunk_begin; y<chunk_end; y++) {
        uint64_t from  = graph_t._index[y];
        uint64_t to    = graph_t._index[y+1];
        for (uint64_t idx=from; idx<to; idx++) {
          uint32_t x = graph_t._edges[idx];
          double contrib = src[x];
          next_val[y] += contrib;
        }
      }
    }
  }
}

double my_abs(double val) {
  if(val >= 0) {
    return val;
  } else {
    return -val;
  }
}

int main(int argc, char* argv[]) {
  int required_level = MPI_THREAD_MULTIPLE;
  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  assert(provided_level >= required_level);
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  g_rank = rank;
  g_nprocs = nprocs;

  assert(numa_available() != -1);
  numa_set_strict(1);
  if (argc < 7) {
    cerr << "Usage: " << argv[0] << " <graph_path> <graph_t_path> <run_mode> <num_iters> <num_threads> <chunk_size>" << endl;
    return -1;
  }
  string graph_path = argv[1];
  string graph_t_path = argv[2];
  string run_mode = argv[3];
  int num_iters = atoi(argv[4]);
  int num_threads = atoi(argv[5]);
  uint32_t chunk_size = atoi(argv[6]);

  if (rank == 0) {
    cout << "  graph_path = " << graph_path << endl;
    cout << "  graph_t_path = " << graph_t_path << endl;
    cout << "  run_mode = " << run_mode << endl;
    cout << "  num_iters = " << num_iters << endl;
    cout << "  num_threads = " << num_threads << endl;
  }

  int run_mode_i;
  if (run_mode == "push") {
    assert(nprocs == 1);
    run_mode_i = RUN_MODE_PUSH;
  } else if (run_mode == "pull") {
    assert(nprocs == 1);
    run_mode_i = RUN_MODE_PULL;
  } else if (run_mode == "delegation_pwq") {
    run_mode_i = RUN_MODE_DELEGATION_PWQ;
  } else {
    cerr << "run mode must be: push, pull, delegation_pwq" << endl;
    return -1;
  }

  // omp_set_num_threads(num_threads);

  Graph<uint32_t, uint64_t> graph;
  graph.load_csr(graph_path);
  Graph<uint32_t, uint64_t> graph_t;
  graph_t.load_csr(graph_t_path);
  assert(graph._num_nodes == graph_t._num_nodes);
  assert(graph._num_edges == graph_t._num_edges);

  if (rank == 0) {
    cout << "  graph_num_nodes = " << graph._num_nodes << endl;
    cout << "  graph_num_edges = " << graph._num_edges << endl;
  }

  size_t num_nodes = graph.num_local_nodes();

  double* src = (double*) malloc_pinned(num_nodes * sizeof(double));
  double* next_val = (double*) malloc_pinned(num_nodes * sizeof(double));
  // double* next_val_1 = (double*) malloc_pinned(num_nodes * sizeof(double));

  for (size_t i=0; i<num_nodes; i++) {
    src[i] = 0.0;
    //next_val[i] = 0.0;
    //next_val_1[i] = 0.0;
    next_val[i] = 1.0 - alpha;
    // next_val_1[i] = 1.0 - alpha;
  }

  if (run_mode_i == RUN_MODE_DELEGATION_PWQ) {
    LaunchConfig launch_config;
    launch_config.load_from_config_file("launch.conf");
    launch_config.distributed_round_robin_socket(rank, g_num_sockets);
    if (rank == 0) {
      launch_config.show();
    }
    // interleave to updater's socket
    // interleave_memory(src, num_nodes * sizeof(double), 4096, launch_config.updater_socket_list, launch_config.num_updater_sockets);
    // interleave_memory(next_val, num_nodes * sizeof(double), 4096, launch_config.updater_socket_list, launch_config.num_updater_sockets);
    AM_Init(launch_config);
  }

  double sum_duration = 0.0;
  uint64_t sum_kernel_duration = 0.0;

  for (int iter=0; iter<num_iters; iter++) {
    uint64_t duration = -currentTimeUs();
    
    #pragma omp parallel for // schedule(dynamic, chunk_size)
    for(uint32_t i=0; i<num_nodes; i++) {
      src[i] = alpha * next_val[i] / (double)(graph.get_index_from_llid(i+1) - graph.get_index_from_llid(i));
      next_val[i] = 1.0 - alpha;
      // next_val_1[i] = 1.0 - alpha;
    }

    if (run_mode_i == RUN_MODE_PUSH) {
      RunPageRankPush(graph, src, next_val, chunk_size);
    } else if (run_mode_i == RUN_MODE_PULL) {
      RunPageRankPull(graph_t, src, next_val, chunk_size);
    } else if (run_mode_i == RUN_MODE_DELEGATION_PWQ) {
      LaunchConfig launch_config;
      launch_config.load_from_config_file("launch.conf");
      launch_config.distributed_round_robin_socket(rank, g_num_sockets);
      // launch_config.load_from_config_file("launch.conf");
      edge_map(launch_config, graph, src, next_val, chunk_size, [](double* next_val, double contrib){*next_val += contrib;});
    }

    MPI_Barrier(MPI_COMM_WORLD);
    duration += currentTimeUs();

    double sum = 0.0;
    for (size_t i=0; i<num_nodes; i++) {
      sum += next_val[i];
    }
    double global_sum;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

    sum_duration += (1e-6 * duration);
    if (rank == 0) {
      cout << "Iteration " << iter << " sum=" << setprecision(14) << global_sum << " duration=" << setprecision(6) << (1e-6 * duration) << "sec" << endl; 
    }
  }

  if (rank == 0) {
    cout << "average duration = " << (sum_duration / num_iters) << endl;
    if (run_mode_i == RUN_MODE_DELEGATION_PWQ) {
      cout << "average kernel_duration = " << (1e-6 * sum_kernel_duration / num_iters) << endl;
    }
  }

  MPI_Finalize();
  return 0;
}
