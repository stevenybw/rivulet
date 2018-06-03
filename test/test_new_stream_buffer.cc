// Simpler PageRank: no fault tolerant, single machine.

#include <algorithm>
#include <condition_variable>
#include <exception>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

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

#include "driver.h"
#include "graph_context.h"
#include "mmap_allocator.h"

using namespace std;

constexpr size_t MPI_SEND_BUFFER_SIZE = 1*1024*1024;
constexpr size_t COMBINE_BUFFER_SIZE  = 128; // Please search StreamingStoreUnrolled64Byte or StreamingStoreUnrolled128Byte
constexpr size_t NUM_BUCKETS          = 1024;

using UpdateRequest = GeneralUpdateRequest<uint32_t, double>;
using LockGuard = std::lock_guard<std::mutex>;
using UniqueLock = std::unique_lock<std::mutex>;

struct WorkerRequest
{
  void* data;
  uint64_t size;
};

template <typename T>
struct ConcurrentQueue
{
  struct closed_error : public exception {
    const char* what() const throw() { 
      return "The queue is closed"; 
    }
  };

  std::mutex mu;
  std::condition_variable cond; // subscribe for queue state change
  volatile bool closed = false;
  std::queue<T> q;

  /*! \brief Push an element to the queue
   */
  void push(const T& e) {
    UniqueLock lk(mu);
    q.push(e);
    lk.unlock();
    cond.notify_one();
  }

  /*! \brief The size of the queue
   */
  size_t size() {
    LockGuard lk(mu);
    return q.size();
  }

  /*! \brief Close the concurrent queue, anyone who pulls from it will get a closed_error.
   */
  void close() {
    UniqueLock lk(mu);
    closed = true;
    lk.unlock();
    cond.notify_all();
  }

  /*! \brief Rearm the concurrent queue, reverse the effect of close.
   */
  void rearm() {
    UniqueLock lk(mu);
    closed = false;
    lk.unlock();
    cond.notify_all();
  }

  /*! \brief Pull an element from the queue. closed_error will be thrown if the queue has been closed
   */
  T pull() {
    UniqueLock lk(mu);
    cond.wait(lk, [this]{ return (!q.empty()) || closed; });
    if (!q.empty()) {
      T ret = q.front();
      q.pop();
      lk.unlock();
      return ret;
    } else {
      assert(closed);
      lk.unlock();
      throw closed_error();
    }
  }
};

/*! \brief In-Memory Buffer Fragment
 *
 *  NUM_BUCKETS BufferFragments consist of a BufferFrame. Each worker has a BufferFrame.
 */
struct BufferFragment { char d[MPI_SEND_BUFFER_SIZE]; };

struct VertexUpdater
{
  using ConcurrentWRQueue = ConcurrentQueue<WorkerRequest>;
  using Thread = std::thread;
  using ThreadList = std::vector<std::thread>;

  MmapMemoryPool* mmap_pool = NULL;
  double* next_val = NULL;

  int num_workers;
  int num_producers;
  int num_buffers;

  ConcurrentWRQueue  responds;
  ConcurrentWRQueue* workers_requests; // [num_workers]

  bool started = false;
  ThreadList workers;

  std::mutex mu_num_worker_done;
  std::condition_variable cond_num_worker_done;
  volatile int num_worker_done = 0;

  /*! \brief Pass driver to create in nvdimm pool
   */
  VertexUpdater(int num_workers, int num_producers, bool enable_nvdimm=false) : num_workers(num_workers), num_producers(num_producers) {
    num_buffers = (num_workers + num_producers) * NUM_BUCKETS; // num buffers in buffer pool
    workers_requests = new ConcurrentWRQueue[num_workers];
    workers.resize(num_workers);
    if (enable_nvdimm) {
      Configuration env_config;
      mmap_pool = new MmapMemoryPool(env_config.nvm_off_cache_pool_dir + "/temp.file", num_buffers * sizeof(BufferFragment));
    }
    for(int i=0; i<num_buffers; i++) {
      void* data = NULL;
      if (enable_nvdimm) {
        data = mmap_pool->alloc<char>(sizeof(BufferFragment), 4096);
      } else {
        data = memalign(4096, sizeof(BufferFragment));
      }
      uint64_t size = 0;
      responds.push({(UpdateRequest*)data, size});
    }
    printf("Total memory consumption for buffers: %.2lf GB\n", 1e-9 * num_buffers * sizeof(BufferFragment));
  }

  void set_next_val(double* next_val) { this->next_val = next_val; }

  /*! \brief Start the vertex updater service
   */
  void start() {
    assert(!started);
    started = true;
    num_worker_done = 0;
    for(int i=0; i<num_workers; i++) {
      workers_requests[i].rearm();
      workers[i] = std::move(Thread([this, i]{ this->_worker(i); }));
    }
  }

  /*! \brief Fetch one buffer from buffer pool
   *
   *  Vertex updater has a pool of BufferFragment (num_workers + num_producers) * NUM_BUCKETS * sizeof(BufferFragment)
   *  This will fetch a BufferFragment from the pool. Blocked if not enough BufferFragment. (TODO we should reduce this
   *  situation)
   */
  void* fetch_one() {
    return responds.pull().data;
  }

  /*! \brief Give back one buffer to buffer pool
   *
   *  Inverse to fetch_one
   */
  void restitution(void* ptr) {
    responds.push({ptr, 0LL});
  }

  void submit(int partition_id, WorkerRequest wr) {
    int worker_id = partition_id % num_workers;
    workers_requests[worker_id].push(wr);
  }

  /*! \brief Close all the worker queue, and wait for all the workers to be done
   *  One thread once only!
   */
  void drain_and_join() {
    assert(started);
    started = false;
    for (int i=0; i<num_workers; i++) {
      workers_requests[i].close();
    }
    UniqueLock lk(mu_num_worker_done);
    cond_num_worker_done.wait(lk, [this]{ return num_worker_done == num_workers; });
    lk.unlock();
    printf("Flush complete\n");
    for(int i=0; i<num_workers; i++) {
      workers[i].join();
    }
    if (responds.size() != num_buffers) {
      printf("Unexpected buffer size: expected %d   current %zu\n", num_buffers, responds.size());
      assert(false);
    }
  }

  void _worker(int worker_id) {
    uint64_t edges_processed = 0;
    uint64_t duration = -currentTimeUs();
    uint64_t active_duration = 0;
    try {
      assert(worker_id>=0 && worker_id < num_workers);
      ConcurrentWRQueue& requests = workers_requests[worker_id];
      double temp = 0.0;
      while (true) {
        WorkerRequest req = requests.pull();
        UpdateRequest* reqs = (UpdateRequest*) req.data;
        uint64_t       size = req.size;
        edges_processed += size;
        active_duration -= currentTimeUs();
        for(uint64_t i=0; i<size; i++) {
          uint32_t y = reqs[i].y;
          double contrib = reqs[i].contrib;
          next_val[y] += contrib;
          _mm_prefetch(&reqs[i+256], _MM_HINT_T1); // software prefetch greatly improves the performance
        }
        active_duration += currentTimeUs();
        req.size = 0;
        responds.push(req);
      }
      printf("%lf\n", temp);
    } catch (ConcurrentWRQueue::closed_error ex) {
      int curr_num_worker_done = 0;
      UniqueLock lk(mu_num_worker_done);
      curr_num_worker_done = ++num_worker_done;
      lk.unlock();
      if (curr_num_worker_done == num_workers) {
        cond_num_worker_done.notify_one(); // only one thread may be allowed to call drain_and_join
      }
      duration += currentTimeUs();
      double active_gpeps = 1e-3 * edges_processed / active_duration;
      printf("[Worker Thread %d] Done edges_processed  %lu   Active Throughput  %lf GPEPS\n", worker_id, edges_processed, active_gpeps);
      return;
    }
  }
};

struct alignas(64) WorkAssignment {
  struct Result {
    uint64_t from_pos;
    uint64_t to_pos;

    bool empty() { return from_pos == to_pos; }
    uint64_t from() { return from_pos; }
    uint64_t to() { return to_pos; }
  };
  uint64_t from_pos;
  uint64_t to_pos;
  volatile uint64_t current_pos;

  void reset() {
    current_pos = from_pos;
  }

  Result fetch(uint64_t num_elem) {
    uint64_t from = __sync_fetch_and_add(&current_pos, num_elem);
    if (from >= to_pos) {
      return {to_pos, to_pos};
    }
    uint64_t to = from + num_elem;
    if (to >= to_pos) {
      to = to_pos;
    }
    return {from, to};
  }
};


/*! \brief Assign vertex in chunk to producers
 *
 *  Try to make each chunk equal
 */
void assign(uint64_t num_vertex, const uint64_t* index, int num_producers, WorkAssignment* assignments)
{
  assert(sizeof(WorkAssignment) == 64);
  uint64_t total_num_edges = index[num_vertex];
  uint64_t edges_per_chunk = total_num_edges / num_producers;
  uint64_t last_vid = 0;
  for(int i=0; i<num_producers-1; i++) {
    uint64_t curr_vid = last_vid;
    uint64_t expect_edge_sum = (i+1) * edges_per_chunk;
    while(curr_vid<num_vertex && index[curr_vid]<expect_edge_sum) {
      curr_vid++;
    }
    assignments[i].from_pos = last_vid;
    assignments[i].to_pos = curr_vid;
    assignments[i].current_pos = last_vid;
    last_vid = curr_vid;
  }
  assignments[num_producers-1].from_pos = last_vid;
  assignments[num_producers-1].to_pos = num_vertex;
  assignments[num_producers-1].current_pos = last_vid;
}

void assign_equal_vertex(uint64_t num_vertex, const uint64_t* index, int num_producers, WorkAssignment* assignments)
{
  uint64_t chunk_size = num_vertex / num_producers;
  for(int i=0; i<num_producers-1; i++) {
    assignments[i].from_pos = i*chunk_size;
    assignments[i].to_pos = (i+1)*chunk_size;
    assignments[i].current_pos = i*chunk_size;
  }
  assignments[num_producers-1].from_pos = (num_producers-1)*chunk_size;
  assignments[num_producers-1].to_pos = num_vertex;
  assignments[num_producers-1].current_pos = (num_producers-1)*chunk_size;
}

int main(int argc, char* argv[]) {
  uint64_t duration;
  RV_Init();
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  CommandLine cmdline(argc, argv, 7, {"num_workers", "graph_path", "num_iters", "enable_nvdimm", "chunk_size", "thread_local_memory_pool_alignment", "run_mode"}, {nullptr, nullptr, nullptr, "false", "512", "4k", "push"});
  int num_workers     = atoi(cmdline.getValue(0));
  string graph_path   = cmdline.getValue(1);
  int num_iters       = atoi(cmdline.getValue(2));
  bool enable_nvdimm = ("true"==string(cmdline.getValue(3))?true:false);
  uint32_t chunk_size = atoi(cmdline.getValue(4));
  size_t thread_local_memory_pool_alignment = convert_unit(string(cmdline.getValue(5)));
  string run_mode     = cmdline.getValue(6);

  if (rank == 0) {
    cout << "  num_workers = " << num_workers << endl;
    cout << "  graph_path = " << graph_path << endl;
    cout << "  num_iters = " << num_iters << endl;
    cout << "  enable_nvdimm = " << enable_nvdimm << endl;
    cout << "  chunk_size = " << chunk_size << endl;
    cout << "  thread_local_memory_pool_alignment = " << thread_local_memory_pool_alignment << endl;
  }

  Configuration env_config;
  ExecutionContext ctx(env_config.nvm_off_cache_pool_dir, env_config.nvm_off_cache_pool_dir, env_config.nvm_on_cahce_pool_dir, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);
  REGION_BEGIN();
  GArray<uint32_t>* edges = driver->create_array<uint32_t>(ObjectRequirement::load_from(graph_path + ".edges", false, 1));
  GArray<uint64_t>* index = driver->create_array<uint64_t>(ObjectRequirement::load_from(graph_path + ".index", false, 1));
  DistributedGraph<uint32_t, uint64_t, partition_id_bits> graph(MPI_COMM_WORLD, edges, index);
  REGION_END("Graph Load");

  int num_threads = 0;
  #pragma omp parallel reduction(+:num_threads)
  {
    num_threads++;
  }

  if (rank == 0) {
    printf("  num_threads = %d\n", num_threads);
  }

  if (rank == 0) {
    printf("  total_num_nodes = %lu\n", graph.total_num_nodes());
    printf("  total_num_edges = %lu\n", graph.total_num_edges());
  }

  size_t num_nodes = graph.local_num_nodes();
  WorkAssignment* assignments = (WorkAssignment*) memalign(64, num_threads * sizeof(WorkAssignment));
  assign(num_nodes, index->data(), num_threads, assignments);
  {
    uint64_t sum_nodes = 0;
    for(int i=0; i<num_threads; i++) {
      uint64_t from_pos = assignments[i].from_pos;
      uint64_t to_pos = assignments[i].to_pos;
      printf("thread %d   from_vid %lu   to_vid %lu  num_edges %lu\n", i, from_pos, to_pos, index->data()[to_pos] - index->data()[from_pos]);
      sum_nodes += to_pos - from_pos;
    }
    assert(sum_nodes == num_nodes);
  }

  LINES;
  double* src = (double*) malloc_pinned(num_nodes * sizeof(double));
  double* next_val = (double*) malloc_pinned(num_nodes * sizeof(double));
  LINES;

  for (size_t i=0; i<num_nodes; i++) {
    src[i] = 0.0;
    next_val[i] = 1.0 - alpha;
  }

  double sum_duration = 0.0;

  VertexUpdater vertex_updater(num_workers, num_threads, enable_nvdimm);
  vertex_updater.set_next_val(next_val);

  for (int iter=0; iter<num_iters; iter++) {
    for(int i=0; i<num_threads; i++) {
      assignments[i].reset();
    }
    vertex_updater.start();

    #pragma omp parallel for
    for(uint32_t i=0; i<num_nodes; i++) {
      src[i] = alpha * next_val[i] / (double)(graph.get_index_from_lid(i+1) - graph.get_index_from_lid(i));
      next_val[i] = 1.0 - alpha;
    }

    uint64_t vertices_processed_per_thread[num_threads];
    uint64_t edges_processed_per_thread[num_threads];
    uint64_t client_us_per_thread[num_threads];

    uint64_t wallclock_duration = -currentTimeUs();
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      thread_local ThreadLocalMemoryPool tl_mp(2*NUM_BUCKETS*(sizeof(int) + COMBINE_BUFFER_SIZE) + 128, thread_local_memory_pool_alignment);
      char* sendbuf[NUM_BUCKETS];
      tl_mp.reset();
      int* curr_bytes_list = tl_mp.alloc<int>(NUM_BUCKETS);
      // char* combinebuf[NUM_BUCKETS]; // write-combine buffer
      // for (int p=0; p<NUM_BUCKETS; p++) {
      //   combinebuf[p] = tl_mp.alloc<char>(COMBINE_BUFFER_SIZE, COMBINE_BUFFER_SIZE); // allocate combinebuf
      // }
      char* combinebuffer = tl_mp.alloc<char>(COMBINE_BUFFER_SIZE, COMBINE_BUFFER_SIZE); 
      for (int p=0; p<NUM_BUCKETS; p++) {
        curr_bytes_list[p] = 0;
        sendbuf[p] = (char*) vertex_updater.fetch_one();
        assert(MPI_SEND_BUFFER_SIZE % COMBINE_BUFFER_SIZE == 0); // MPI Buffer must be a multiple of combine buffer
      }

      // number of edges processed
      uint64_t vertices_processed = 0;
      uint64_t edges_processed = 0;
      uint64_t duration = -currentTimeUs();
      uint64_t wait_duration = 0;

      int curr_id = tid;
      int num_visited = 0;
      while (num_visited < num_threads) {
        auto work = assignments[curr_id].fetch(chunk_size);
        if (work.empty()) {
          curr_id = (curr_id + 1) % num_threads;
          num_visited++;
          continue;
        }
        size_t from_vid = work.from();
        size_t to_vid   = work.to();
        for (size_t u=from_vid; u<to_vid; u++) {
          vertices_processed++;
          uint64_t from  = graph.get_index_from_lid(u);
          uint64_t to    = graph.get_index_from_lid(u+1);
          edges_processed += (to-from);
          double contrib = src[u];
          for (uint64_t idx=from; idx<to; idx++) {
            uint32_t vid = graph.get_edge_from_index(idx);
            int vid_part = vid >> 17; // the partition this vertex belongs to
            size_t curr_bytes = curr_bytes_list[vid_part];
#if 1
            // check to see if a write-back (and corresponding issue) is required
            if (UNLIKELY(curr_bytes > 0 && curr_bytes % COMBINE_BUFFER_SIZE == 0)) {
              // where to write back
              size_t write_back_pos = curr_bytes - COMBINE_BUFFER_SIZE;
              // first write back into memory buffer
              Memcpy<UpdateRequest>::StreamingStoreUnrolled128Byte(&sendbuf[vid_part][write_back_pos], (UpdateRequest*) &combinebuffer[vid_part*COMBINE_BUFFER_SIZE] , COMBINE_BUFFER_SIZE/sizeof(UpdateRequest));
              // then send out if required
              if (UNLIKELY(curr_bytes == MPI_SEND_BUFFER_SIZE)) {
                // Submit sendbuf[vid_part] of size curr_bytes, and fetch a new buffer

                wait_duration = -currentTimeUs();
                vertex_updater.submit(vid_part, {sendbuf[vid_part], curr_bytes/sizeof(UpdateRequest)});
                sendbuf[vid_part] = (char*) vertex_updater.fetch_one();
                wait_duration += currentTimeUs();

                curr_bytes_list[vid_part] = 0; // reset streaming buffer
                curr_bytes = 0;
              }
            }
#else
            // disable write-back and submit
            if (UNLIKELY(curr_bytes == MPI_SEND_BUFFER_SIZE)) {
              curr_bytes_list[vid_part] = 0; // reset streaming buffer
              curr_bytes = 0;
            }
#endif
            // write into combine buffer
            UpdateRequest* combine = (UpdateRequest*) &combinebuffer[vid_part*COMBINE_BUFFER_SIZE + curr_bytes%COMBINE_BUFFER_SIZE];
            combine->y       = vid;
            combine->contrib = contrib;
            curr_bytes_list[vid_part] = curr_bytes + sizeof(UpdateRequest);
            _mm_prefetch(&graph._edges[idx+128], _MM_HINT_T1);
            // _mm_prefetch(&graph._index[idx+64], _MM_HINT_T2);
          }
          // _mm_prefetch(&graph._index[u+8], _MM_HINT_T1);
        }
      }

#if 1
      // flush buffer
      for (int part_id=0; part_id<NUM_BUCKETS; part_id++) {
        size_t curr_bytes = curr_bytes_list[part_id];
        if (curr_bytes > 0) {
          { // write back
            size_t combine_buffer_bytes = curr_bytes%COMBINE_BUFFER_SIZE;
            if (combine_buffer_bytes==0 && curr_bytes!=0) {
              combine_buffer_bytes = COMBINE_BUFFER_SIZE;
            }
            assert(combine_buffer_bytes % sizeof(UpdateRequest) == 0);
            Memcpy<UpdateRequest>::StreamingStore(&sendbuf[part_id][curr_bytes - combine_buffer_bytes], (UpdateRequest*) &combinebuffer[part_id*COMBINE_BUFFER_SIZE], combine_buffer_bytes/sizeof(UpdateRequest));
          }
          vertex_updater.submit(part_id, {sendbuf[part_id], curr_bytes/sizeof(UpdateRequest)});
          sendbuf[part_id] = (char*) vertex_updater.fetch_one();
          curr_bytes_list[part_id] = 0;
        }
      }
#endif

      duration += currentTimeUs();
      client_us_per_thread[tid] = duration;
      vertices_processed_per_thread[tid] = vertices_processed;
      edges_processed_per_thread[tid] = edges_processed;

      double gpeps = 1e-3 * edges_processed / duration;
      double active_gpeps = 1e-3 * edges_processed / (duration - wait_duration);
      printf("[Client Thread %d/%d] Done  edges_processed  %lu  wall_gpeps %lf GPEPS  active_gpeps %lf GPEPS\n", g_rank, tid, edges_processed, gpeps, active_gpeps);

      for (int p=0; p<NUM_BUCKETS; p++) {
        vertex_updater.restitution((void*) sendbuf[p]);
      }
    }
    wallclock_duration += currentTimeUs();
    // wait for completion
    vertex_updater.drain_and_join();
    uint64_t min_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t)-1, [](uint64_t lhs, uint64_t rhs){return min(lhs, rhs);});
    uint64_t sum_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return lhs+rhs;});
    uint64_t max_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return max(lhs, rhs);});
    double   avg_edges_processed = sum_edges_processed / num_threads;
    uint64_t sum_vertices_processed = accumulate(vertices_processed_per_thread, vertices_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return lhs+rhs;});
    uint64_t total_memory_access = (sizeof(UpdateRequest) + sizeof(uint32_t)) * sum_edges_processed + sizeof(double) * sum_vertices_processed;
    double   gbps  = 1e-3 * total_memory_access/wallclock_duration;
    double   gpeps = 1e-3 * sum_edges_processed/wallclock_duration;
    printf("%10s %10s %10s %10s %10s %10s\n", "", "min_edges", "avg_edges", "max_edges", "GPEPS", "");
    printf("%10s %10lu %10lf %10lu %10lf %10s\n", "Client", min_edges_processed, avg_edges_processed, max_edges_processed, gpeps, "");
    double sum = 0.0;
    for (size_t i=0; i<num_nodes; i++) {
      sum += next_val[i];
    }
    double global_sum;
    MPI_Allreduce(&sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    sum_duration += (1e-6 * wallclock_duration);
    if (rank == 0) {
      cout << "Iteration " << iter << " sum=" << setprecision(14) << global_sum << " duration=" << setprecision(6) << (1e-6 * wallclock_duration) << "sec" << endl; 
    }
  }

  if (rank == 0) {
    cout << "average duration = " << (sum_duration / num_iters) << endl;
  }

  RV_Finalize();
  return 0;
}
