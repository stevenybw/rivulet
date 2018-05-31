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

#include "driver.h"
#include "graph_context.h"

using namespace std;

// struct PageRankPushUpdater {
//   double* curr_val;
//   double* next_val;
//   double vertex_value(VertexId u) override { return curr_val[u]; }
//   void process_update(VertexId v, double contrib) { Atomic<double>::cas_atomic_update(&next_val[v], contrib); }
// };

// template <typename VertexId>
// struct PageRankProgram : public GraphProgram<VertexId, double> {
//   double* current_val;
//   double* next_val;
// 
//   PageRankProgram(double* current_val, double* next_val) : current_val(current_val), next_val(next_val) {}
// 
//   // get local accumulation from mirror v, given edges it
//   void dense_signal(VertexId v, InEdgesIterator it) override {
//     double sum = 0.0;
//     for(InEdge edge : it) {
//       VertexId src = edge.neighbor();
//       sum += curr_val[src];
//     }
//     this->emit(v, sum);
//   }
// 
//   // master v receives a message msg
//   VertexId dense_slot(VertexId v, double msg) override {
//     Atomic<double>::cas_atomic_update(&next_val[v], contrib);
//     return 1;
//   }
// };
// 
// struct PageRankPullUpdater {
//   double* curr_val;
//   double* next_val;
//   VertexSubset active_verteces();
//   double vertex_value(VertexId u) override { return curr_val[u]; }
// };

const char* run_mode_push = "push";
const char* run_mode_pull = "pull";

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

  if (rank == 0) {
    printf("  total_num_nodes = %llu\n", graph.total_num_nodes());
    printf("  total_num_edges = %llu\n", graph.total_num_edges());
  }

  LaunchConfig launch_config;
  launch_config.load_from_config_file("launch.conf");
  // launch_config.distributed_round_robin_socket(rank, g_num_sockets);
  if (rank == 0) {
    launch_config.show();
  }
  GraphContext graph_context(launch_config);

  size_t num_nodes = graph.local_num_nodes();
  LINES;
  double* src = (double*) malloc_pinned(num_nodes * sizeof(double));
  double* next_val = (double*) malloc_pinned(num_nodes * sizeof(double));
  LINES;

  uint64_t global_empty_node = 0;
  {
    double* in_degree = src;
    double* out_degree = next_val;

    #pragma omp parallel for
    for(int i=0; i<num_nodes; i++) {
      in_degree[i] = 0.0;
      out_degree[i] = (double) graph.get_out_degree(i);
    }

    VertexRange<uint32_t> all_vertex(0, num_nodes);
    auto vertex_value_op = [](uint32_t u) { return 1.0; };
    auto on_update_op = [in_degree](uint32_t v, double value) { in_degree[v] += 1; };
    auto on_update_gen = [&on_update_op]() { return on_update_op; };
    graph_context.compute_push_delegate<double, decltype(on_update_op)>(graph, all_vertex.begin(), all_vertex.end(), vertex_value_op, on_update_gen, chunk_size);
    
    uint64_t empty_nodes = 0;
    #pragma omp parallel for reduction(+: empty_nodes)
    for(int i=0; i<num_nodes; i++) {
      if (in_degree[i] == 0 && out_degree[i] == 0) {
        empty_nodes++;
      }
    }
    MPI_Allreduce(&empty_nodes, &global_empty_node, 1, MPI_UNSIGNED_LONG, MPI_SUM, MPI_COMM_WORLD);
    if (rank == 0) {
      printf("total number of empty nodes (in deg & out deg both are zero): %lu\n", global_empty_node);
    }
  }

  for (size_t i=0; i<num_nodes; i++) {
    src[i] = 0.0;
    next_val[i] = 1.0 - alpha;
  }

  double sum_duration = 0.0;

  for (int iter=0; iter<num_iters; iter++) {
    uint64_t duration = -currentTimeUs();
    
    #pragma omp parallel for // schedule(dynamic, chunk_size)
    for(uint32_t i=0; i<num_nodes; i++) {
      src[i] = alpha * next_val[i] / (double)(graph.get_index_from_lid(i+1) - graph.get_index_from_lid(i));
      next_val[i] = 1.0 - alpha;
    }

    if (run_mode == run_mode_push) {
      // graph_context.edge_map(graph, src, next_val, chunk_size, [](double* next_val, double contrib){*next_val += contrib;});
      VertexRange<uint32_t> all_vertex(0, num_nodes);
      auto vertex_value_op = [src](uint32_t u){return src[u];};
      auto on_update_op = [next_val](uint32_t v, double value) { next_val[v] += value; };
      auto on_update_gen = [&on_update_op]() {return on_update_op;};
      graph_context.compute_push_delegate<double, decltype(on_update_op)>(graph, all_vertex.begin(), all_vertex.end(), vertex_value_op, on_update_gen, chunk_size);
    } else if (run_mode == run_mode_pull) {
      assert(false); // TODO: Implement Pull
      // graph_context.compute_pull<PageRankProgram>();
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
      cout << "Iteration " << iter << " sum=" << setprecision(14) << global_sum << " sum (without empty vertex)=" << setprecision(14) << (global_sum - global_empty_node*(1.0-alpha))  << " duration=" << setprecision(6) << (1e-6 * duration) << "sec" << endl; 
    }
  }

  if (rank == 0) {
    cout << "average duration = " << (sum_duration / num_iters) << endl;
  }

  RV_Finalize();
  return 0;
}
