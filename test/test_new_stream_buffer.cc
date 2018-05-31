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

using namespace std;

constexpr size_t MPI_SEND_BUFFER_SIZE = 1*1024*1024;
constexpr size_t COMBINE_BUFFER_SIZE  = 128;
constexpr size_t NUM_BUCKETS          = 512;

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

  VertexUpdater(int num_workers, int num_producers) : num_workers(num_workers), num_producers(num_producers) {
    num_buffers = (num_workers + num_producers) * NUM_BUCKETS; // num buffers in buffer pool
    workers_requests = new ConcurrentWRQueue[num_workers];
    workers.resize(num_workers);
    for(int i=0; i<num_buffers; i++) {
      void* data = memalign(4096, sizeof(BufferFragment));
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
      printf("Unexpected buffer size: expected %d   current %d\n", num_buffers, responds.size());
      assert(false);
    }
  }

  void _worker(int worker_id) {
    try {
      assert(worker_id>=0 && worker_id < num_workers);
      ConcurrentWRQueue& requests = workers_requests[worker_id];
      while (true) {
        WorkerRequest req = requests.pull();
        UpdateRequest* reqs = (UpdateRequest*) req.data;
        uint64_t       size = req.size;
        for(uint64_t i=0; i<size; i++) {
          uint32_t y = reqs[i].y;
          double contrib = reqs[i].contrib;
          next_val[y] += contrib;
        }
        req.size = 0;
        responds.push(req);
      }
    } catch (ConcurrentWRQueue::closed_error ex) {
      int curr_num_worker_done = 0;
      UniqueLock lk(mu_num_worker_done);
      curr_num_worker_done = ++num_worker_done;
      lk.unlock();
      if (curr_num_worker_done == num_workers) {
        cond_num_worker_done.notify_one(); // only one thread may be allowed to call drain_and_join
      }
      printf("worker %d closed\n", worker_id);
      return;
    }
  }
};

int main(int argc, char* argv[]) {
  uint64_t duration;
  RV_Init();
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  CommandLine cmdline(argc, argv, 5, {"num_workers", "graph_path", "num_iters", "chunk_size", "run_mode"}, {nullptr, nullptr, nullptr, "512", "push"});
  int num_workers     = atoi(cmdline.getValue(0));
  string graph_path   = cmdline.getValue(1);
  int num_iters       = atoi(cmdline.getValue(2));
  uint32_t chunk_size = atoi(cmdline.getValue(3));
  string run_mode     = cmdline.getValue(4);

  if (rank == 0) {
    cout << "  num_workers = " << num_workers << endl;
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
  LINES;
  double* src = (double*) malloc_pinned(num_nodes * sizeof(double));
  double* next_val = (double*) malloc_pinned(num_nodes * sizeof(double));
  LINES;

  for (size_t i=0; i<num_nodes; i++) {
    src[i] = 0.0;
    next_val[i] = 1.0 - alpha;
  }

  double sum_duration = 0.0;

  VertexUpdater vertex_updater(num_workers, num_threads);
  vertex_updater.set_next_val(next_val);

  for (int iter=0; iter<num_iters; iter++) {
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
      thread_local ThreadLocalMemoryPool tl_mp(2*NUM_BUCKETS*(sizeof(int) + COMBINE_BUFFER_SIZE) + 128);
      char* sendbuf[NUM_BUCKETS];
      tl_mp.reset();
      int* curr_bytes_list = tl_mp.alloc<int>(NUM_BUCKETS);
      char* combinebuf[NUM_BUCKETS]; // write-combine buffer
      for (int p=0; p<NUM_BUCKETS; p++) {
        combinebuf[p] = tl_mp.alloc<char>(COMBINE_BUFFER_SIZE, COMBINE_BUFFER_SIZE); // allocate combinebuf
      }
      for (int p=0; p<NUM_BUCKETS; p++) {
        curr_bytes_list[p] = 0;
        sendbuf[p] = (char*) vertex_updater.fetch_one();
        assert(MPI_SEND_BUFFER_SIZE % COMBINE_BUFFER_SIZE == 0); // MPI Buffer must be a multiple of combine buffer
      }

      // number of edges processed
      uint64_t vertices_processed = 0;
      uint64_t edges_processed = 0;
      uint64_t duration = -currentTimeUs();

      #pragma omp for schedule(dynamic, chunk_size)
      for (size_t u=0; u<num_nodes; u++) {
        vertices_processed++;
        uint64_t from  = graph.get_index_from_lid(u);
        uint64_t to    = graph.get_index_from_lid(u+1);
        edges_processed += (to-from);
        double contrib = src[u];
        for (uint64_t idx=from; idx<to; idx++) {
          uint32_t vid = graph.get_edge_from_index(idx);
          int vid_part = (vid >> 5) % NUM_BUCKETS; // the partition this vertex belongs to
          size_t curr_bytes = curr_bytes_list[vid_part];
          // check to see if a write-back (and corresponding issue) is required
          if (curr_bytes > 0 && curr_bytes % COMBINE_BUFFER_SIZE == 0) {
            // where to write back
            size_t write_back_pos = curr_bytes - COMBINE_BUFFER_SIZE;
            // first write back into memory buffer
            Memcpy<UpdateRequest>::StreamingStoreUnrolled(&sendbuf[vid_part][write_back_pos], (UpdateRequest*) &combinebuf[vid_part][0] , COMBINE_BUFFER_SIZE/sizeof(UpdateRequest));
            // then send out if required
            if (curr_bytes == MPI_SEND_BUFFER_SIZE) {
              // Submit sendbuf[vid_part] of size curr_bytes, and fetch a new buffer
              vertex_updater.submit(vid_part, {sendbuf[vid_part], curr_bytes/sizeof(UpdateRequest)});
              sendbuf[vid_part] = (char*) vertex_updater.fetch_one();
              curr_bytes_list[vid_part] = 0; // reset streaming buffer
              curr_bytes = 0;
            }
            // Memcpy<UpdateRequest>::PrefetchNTA(req_ptr+1);
          }
          // write into combine buffer
          UpdateRequest* combine = (UpdateRequest*) &combinebuf[vid_part][curr_bytes % COMBINE_BUFFER_SIZE];
          combine->y       = vid;
          combine->contrib = contrib;
          curr_bytes_list[vid_part] = curr_bytes + sizeof(UpdateRequest);
        }
      }

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
            Memcpy<UpdateRequest>::StreamingStore(&sendbuf[part_id][curr_bytes - combine_buffer_bytes], (UpdateRequest*) &combinebuf[part_id][0], combine_buffer_bytes/sizeof(UpdateRequest));
          }
          vertex_updater.submit(part_id, {sendbuf[part_id], curr_bytes/sizeof(UpdateRequest)});
          sendbuf[part_id] = (char*) vertex_updater.fetch_one();
          curr_bytes_list[part_id] = 0;
        }
      }

      duration += currentTimeUs();
      client_us_per_thread[tid] = duration;
      vertices_processed_per_thread[tid] = vertices_processed;
      edges_processed_per_thread[tid] = edges_processed;
      printf("[Client Thread %d/%d] Done  edges_processed   %lu\n", g_rank, tid, edges_processed);

      for (int p=0; p<NUM_BUCKETS; p++) {
        vertex_updater.restitution((void*) sendbuf[p]);
      }
    }
    // wait for completion
    vertex_updater.drain_and_join();

    wallclock_duration += currentTimeUs();
    uint64_t min_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t)-1, [](uint64_t lhs, uint64_t rhs){return min(lhs, rhs);});
    uint64_t sum_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return lhs+rhs;});
    uint64_t max_edges_processed = accumulate(edges_processed_per_thread, edges_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return max(lhs, rhs);});
    double   avg_edges_processed = sum_edges_processed / num_threads;
    uint64_t sum_vertices_processed = accumulate(vertices_processed_per_thread, vertices_processed_per_thread + num_threads, (uint64_t) 0, [](uint64_t lhs, uint64_t rhs){return lhs+rhs;});
    uint64_t total_memory_access = (sizeof(UpdateRequest) + sizeof(uint32_t)) * sum_edges_processed + sizeof(double) * sum_vertices_processed;
    double   gpeps = 1e-3 * sum_edges_processed/wallclock_duration;
    printf("%10s %10s %10s %10s %10s %10s\n", "", "min_edges", "avg_edges", "max_edges", "GB/s", "GPEPS");
    printf("%10s %10lu %10lf %10lu %10lf %10lf\n", "Client", min_edges_processed, avg_edges_processed, max_edges_processed, 1.0e-3 * total_memory_access/wallclock_duration, gpeps);
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
