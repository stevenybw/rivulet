#include <iostream>

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "kmeans_model.h"
#include "driver.h"

using namespace std;


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
  printf("nprocs: %d\n", nprocs);
  if (argc < 6) {
    cerr << "Usage: " << argv[0] << " <input_data_path in NFS> <dimension of samples> <numClusters> <numIterations> <anonymous_prefix>" << endl;
    return -1;
  }

  string data_path = argv[1];
  int dim = atoi(argv[2]);
  int k = atoi(argv[3]);
  int ite = atoi(argv[4]);
  string anonymous_prefix = argv[5];

  ExecutionContext ctx(anonymous_prefix, anonymous_prefix, anonymous_prefix, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);
  LOG_BEGIN();
  LOG_INFO("Loading kmeans data");
  GArray<float>* kmeans_data = driver->readFromBinaryRecords<float>(data_path, dim*sizeof(float));
  KMeansModel km = KMeansModel(ctx);
  LOG_INFO("Train");
  km.train(kmeans_data, dim, k, ite);

  float* centers = km.getCenters();
  float cost = km.computeCost();
  if(rank == 0){
    cerr << "within set sum of squared errors: " << cost << endl;
  }
  delete kmeans_data;
  MPI_Finalize();
  return 0;
}
