// Simpler PageRank: no fault tolerant, single machine.

#include <algorithm>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <numeric>
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

#include "driver.h"
#include "graph_context.h"

using namespace std;

constexpr size_t MPI_SEND_BUFFER_SIZE = 1*1024*1024;
constexpr size_t COMBINE_BUFFER_SIZE  = 128;
constexpr size_t NUM_BUCKETS          = 512;

using UpdateRequest = GeneralUpdateRequest<uint32_t, double>;

int main(int argc, char* argv[]) {
  uint64_t duration;
  RV_Init();
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  CommandLine cmdline(argc, argv, 4, {"graph_path", "num_iters", "chunk_size", "run_mode"}, {nullptr, nullptr, "512", "push"});
  string graph_path   = cmdline.getValue(0);
  int num_iters       = atoi(cmdline.getValue(1));
  uint32_t chunk_size = atoi(cmdline.getValue(2));
  string run_mode     = cmdline.getValue(3);

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

  for (int iter=0; iter<num_iters; iter++) {
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
      thread_local ThreadLocalMemoryPool tl_mp(2*NUM_BUCKETS*(sizeof(int) + MPI_SEND_BUFFER_SIZE) + 128 + MPI_SEND_BUFFER_SIZE);
      tl_mp.reset();
      int* buf_id_list = tl_mp.alloc<int>(NUM_BUCKETS);
      int* curr_bytes_list = tl_mp.alloc<int>(NUM_BUCKETS);
      char* combinebuf[NUM_BUCKETS]; // write-combine buffer
      char* sendbuf[NUM_BUCKETS][2];
      for (int p=0; p<NUM_BUCKETS; p++) {
        combinebuf[p] = tl_mp.alloc<char>(COMBINE_BUFFER_SIZE, COMBINE_BUFFER_SIZE); // allocate combinebuf
      }
      for (int p=0; p<NUM_BUCKETS; p++) {
        buf_id_list[p] = 0;
        curr_bytes_list[p] = 0;
        for (int i=0; i<2; i++) {
          sendbuf[p][i] = tl_mp.alloc<char>(MPI_SEND_BUFFER_SIZE, 4096);
          assert(sendbuf[p][i]);
        }
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
            int    buf_id = buf_id_list[vid_part];
            Memcpy<UpdateRequest>::StreamingStoreUnrolled(&sendbuf[vid_part][buf_id][write_back_pos], (UpdateRequest*) &combinebuf[vid_part][0] , COMBINE_BUFFER_SIZE/sizeof(UpdateRequest));
            // then send out if required
            if (curr_bytes == MPI_SEND_BUFFER_SIZE) {
              int flag = 0;
              // Process sendbuf[vid_part][buf_id] of size curr_bytes here
              // MT_MPI_Isend(sendbuf[vid_part][buf_id], curr_bytes, MPI_CHAR, vid_part, TAG_DATA, MPI_COMM_WORLD, &req[vid_part][buf_id]);
              // while (!flag) {
              //   MT_MPI_Test(&req[vid_part][buf_id^1], &flag, MPI_STATUS_IGNORE);
              // }
              buf_id_list[vid_part] = buf_id^1; // switch the buffer
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

      // flush
      for (int part_id=0; part_id<NUM_BUCKETS; part_id++) {
        int    buf_id = buf_id_list[part_id];
        size_t curr_bytes = curr_bytes_list[part_id];
        { // write back
          size_t combine_buffer_bytes = curr_bytes%COMBINE_BUFFER_SIZE;
          if (combine_buffer_bytes==0 && curr_bytes!=0) {
            combine_buffer_bytes = COMBINE_BUFFER_SIZE;
          }
          assert(combine_buffer_bytes % sizeof(UpdateRequest) == 0);
          Memcpy<UpdateRequest>::StreamingStore(&sendbuf[part_id][buf_id][curr_bytes - combine_buffer_bytes], (UpdateRequest*) &combinebuf[part_id][0], combine_buffer_bytes/sizeof(UpdateRequest));
        }        
//        int flag = 0;
//        while (!flag) {
//          MT_MPI_Test(&req[part_id][buf_id], &flag, MPI_STATUS_IGNORE);
//        }
//        req[part_id][buf_id] = MPI_REQUEST_NULL;
//        while (!flag) {
//          MT_MPI_Test(&req[part_id][buf_id^1], &flag, MPI_STATUS_IGNORE);
//        }
//        req[part_id][buf_id^1] = MPI_REQUEST_NULL;
//        MT_MPI_Send(sendbuf[part_id][buf_id], curr_bytes, MPI_CHAR, part_id, TAG_DATA, MPI_COMM_WORLD);
//        MT_MPI_Send(NULL, 0, MPI_CHAR, part_id, TAG_CLOSE, MPI_COMM_WORLD);
        buf_id = buf_id ^ 1;
        curr_bytes = 0;
        curr_bytes_list[part_id] = curr_bytes;
        buf_id_list[part_id] = buf_id;
      }

      duration += currentTimeUs();
      client_us_per_thread[tid] = duration;
      vertices_processed_per_thread[tid] = vertices_processed;
      edges_processed_per_thread[tid] = edges_processed;
      printf("[Client Thread %d/%d] Done  edges_processed   %lu\n", g_rank, tid, edges_processed);
    }
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
