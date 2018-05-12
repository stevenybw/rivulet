#include <iostream>

#include <mpi.h>
#include <omp.h>

#include "driver.h"
#include "graph_context.h"

using namespace std;

const size_t partition_id_bits = 1;

template <typename T>
struct Queue {
  struct Iterator {
    T* data;
    uint64_t pos;
    Iterator(T* data, uint64_t pos) : data(data), pos(pos) {}
    Iterator operator+(uint64_t offset) { return Iterator(data, pos+offset); }
    T& operator*() { return data[pos]; }
    uint64_t operator-(Iterator rhs) { return pos - rhs.pos; }
  };
  constexpr static size_t THREAD_LOCAL_BUFFER_BYTES = 1024;
  constexpr static size_t THREAD_LOCAL_BUFFER_SIZE  = THREAD_LOCAL_BUFFER_BYTES / sizeof(T);
  struct ThreadLocalContext {
    ThreadLocalContext() : num_elem(0) {}
    size_t num_elem;
    T buffer[THREAD_LOCAL_BUFFER_SIZE];
  };

  uint64_t _capacity;
  T* _data;
  uint64_t _size alignas(64);

  Queue (uint64_t capacity) {
    _capacity = capacity;
    _size = 0;
    _data = (T*) malloc_pinned(capacity * sizeof(T));
  }

  void add(const T& e) {
    _data[_size++] = e;
  }

  void add(ThreadLocalContext& ctx, const T& e) {
    if (ctx.num_elem == THREAD_LOCAL_BUFFER_SIZE) {
      uint64_t pos = __sync_fetch_and_add(&_size, THREAD_LOCAL_BUFFER_SIZE);
      memcpy(&_data[pos], ctx.buffer, THREAD_LOCAL_BUFFER_BYTES);
      ctx.num_elem = 0;
    }
    ctx.buffer[ctx.num_elem++] = e;
  }

  void flush(ThreadLocalContext& ctx) {
    uint64_t pos = __sync_fetch_and_add(&_size, ctx.num_elem);
    memcpy(&_data[pos], ctx.buffer, ctx.num_elem * sizeof(T));
    ctx.num_elem = 0;
  }

  void swap(Queue<T>& rhs) {
    ::swap(_capacity, rhs._capacity);
    ::swap(_data, rhs._data);
    ::swap(_size, rhs._size);
  }

  void clear() {
    _size = 0;
  }

  size_t size() {
    return _size;
  }

  Iterator begin() { return Iterator(_data, 0); }
  Iterator end() { return Iterator(_data, _size); }
};

struct OnUpdate
{
  uint32_t* num_delta_active;
  uint32_t* parent;
  Queue<uint32_t>& next_frontier;
  Queue<uint32_t>::ThreadLocalContext next_frontier_tlc;
  uint32_t  local_num_delta_active;

  OnUpdate(uint32_t* num_delta_active, uint32_t* parent, Queue<uint32_t>& next_frontier) : 
        num_delta_active(num_delta_active), 
        parent(parent),
        next_frontier(next_frontier),
        local_num_delta_active(0) { }

  ~OnUpdate() {
    // flush before being destroyed
    __sync_fetch_and_add(num_delta_active, local_num_delta_active);
    next_frontier.flush(next_frontier_tlc);
  }

  void operator()(uint32_t v, uint32_t vertex_value) {
    if (parent[v] == (uint32_t)-1) {
      local_num_delta_active++;
      parent[v] = vertex_value;
      next_frontier.add(next_frontier_tlc, v);
    }
  }
};

int main(int argc, char* argv[])
{
  uint64_t duration;
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
    cerr << "Usage: " << argv[0] << " <graph_path> <num_iters> <chunk_size>" << endl;
    return -1;
  }
  string graph_path = argv[1];
  int num_iters = atoi(argv[2]);
  uint32_t chunk_size = atoi(argv[3]);

  if (rank == 0) {
    cout << "  graph_path = " << graph_path << endl;
    cout << "  num_iters = " << num_iters << endl;
    cout << "  chunk_size = " << chunk_size << endl;
  }

  Configuration env_config;
  ExecutionContext ctx(env_config.nvm_off_cache_pool_dir, env_config.nvm_off_cache_pool_dir, env_config.nvm_on_cahce_pool_dir, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);
  REGION_BEGIN();
  GArray<uint32_t>* edges = driver->create_array<uint32_t>(ObjectRequirement::load_from(graph_path + ".edges"));
  GArray<uint64_t>* index = driver->create_array<uint64_t>(ObjectRequirement::load_from(graph_path + ".index"));
  DistributedGraph<uint32_t, uint64_t, partition_id_bits> graph(MPI_COMM_WORLD, edges, index);
  REGION_END("Graph Load");

  if (rank == 0) {
    printf("  total_num_nodes = %llu\n", graph.total_num_nodes());
    printf("  total_num_edges = %llu\n", graph.total_num_edges());
  }

  size_t local_num_nodes = graph.local_num_nodes();
  uint32_t* parent = (uint32_t*) malloc(local_num_nodes * sizeof(uint32_t));
  Queue<uint32_t> curr_frontier(local_num_nodes);
  Queue<uint32_t> next_frontier(local_num_nodes);
  memset(parent, 0xFF, local_num_nodes * sizeof(uint32_t));

  for(int i=0; i<65536; i++) {
    curr_frontier.add(i);
  }

  LaunchConfig launch_config;
  launch_config.load_from_config_file("launch.conf");
  if (rank == 0) {
    launch_config.show();
  }
  GraphContext graph_context(launch_config);
  uint32_t num_delta_active = 0;
  // u is local vertex id, should be transform to vid, then emit to target
  auto vertex_value = [&graph](uint32_t u) {return graph.get_vid_from_lid(u);};
  // auto on_update = [&num_delta_active, parent](ThreadContext& thread_ctx, uint32_t v, uint32_t vertex_value) {
  //   if (parent[v] == -1) {
  //     num_delta_active++;
  //     parent[v] = vertex_value;
  //   }
  // };
  auto update_generator = [&next_frontier, &num_delta_active, parent]() { return OnUpdate(&num_delta_active, parent, next_frontier); };
  uint32_t iter_id = 0;
  while (true) {
    size_t local_curr_active = curr_frontier.size();
    size_t global_curr_active = 0;
    MPI_Allreduce(&local_curr_active, &global_curr_active, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (global_curr_active == 0) {
      break;
    }
    num_delta_active = 0;
    graph_context.compute_push_delegate<uint32_t, OnUpdate>(graph, curr_frontier.begin(), curr_frontier.end(), vertex_value, update_generator, chunk_size);
    size_t local_next_active = next_frontier.size();
    size_t global_next_active = 0;
    MPI_Allreduce(&local_next_active, &global_next_active, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    uint32_t global_num_delta_active;
    MPI_Allreduce(&num_delta_active, &global_num_delta_active, 1, MPI_UNSIGNED, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("Iter %u size(curr_frontier)=%zu size(next_frontier)=%zu global_num_delta_active=%u\n", iter_id, global_curr_active, global_next_active, global_num_delta_active);
    }
    curr_frontier.swap(next_frontier);
    next_frontier.clear();
    iter_id++;
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();

  return 0; 
}