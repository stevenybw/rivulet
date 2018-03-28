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

#include "graph.h"
#include "graph_context.h"
#include "util.h"

using namespace std;

int main(int argc, char* argv[]) {
  int required_level = MPI_THREAD_MULTIPLE;
  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  assert(provided_level >= required_level);
  init_debug();
  int rank, nprocs;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  if (nprocs == 1) {
    g_group_size = 1;
  } else {
    g_group_size = g_num_sockets;
  }
  g_rank = rank;
  g_nprocs = nprocs;

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

  for (size_t i=0; i<num_nodes; i++) {
    src[i] = 0.0;
    next_val[i] = 1.0 - alpha;
  }

  LaunchConfig launch_config;
  launch_config.load_from_config_file("launch.conf");
  launch_config.distributed_round_robin_socket(rank, g_num_sockets);
  if (rank == 0) {
    launch_config.show();
  }
  GraphContext graph_context(launch_config);

  double sum_duration = 0.0;

  for (int iter=0; iter<num_iters; iter++) {
    uint64_t duration = -currentTimeUs();
    
    #pragma omp parallel for // schedule(dynamic, chunk_size)
    for(uint32_t i=0; i<num_nodes; i++) {
      src[i] = alpha * next_val[i] / (double)(graph.get_index_from_llid(i+1) - graph.get_index_from_llid(i));
      next_val[i] = 1.0 - alpha;
    }

    if (run_mode_i == RUN_MODE_PUSH) {
      graph_context.edge_map_smp_push(graph, src, next_val, chunk_size, [](double* next_val, double contrib){Atomic<double>::cas_atomic_update(next_val, contrib);});
    } else if (run_mode_i == RUN_MODE_PULL) {
      graph_context.edge_map_smp_pull(graph_t, src, next_val, chunk_size, [](double* next_val, double contrib){*next_val += contrib;});
    } else if (run_mode_i == RUN_MODE_DELEGATION_PWQ) {
      graph_context.edge_map(graph, src, next_val, chunk_size, [](double* next_val, double contrib){*next_val += contrib;});
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
  }

  MPI_Finalize();
  return 0;
}
