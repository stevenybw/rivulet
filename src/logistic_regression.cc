#include <iostream>

#include <assert.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#include "driver.h"
#include "logistic_regression_model.h"

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

  if (argc < 5) {
    cerr << "Usage: " << argv[0] << " <training_set_path in NFS> <dimension of simples> "
                                    << "<numIterations> <anonymous_prefix> [test_set_path in NFS]" << endl;
    return -1;
  }
  char *pend;
  string training_path = argv[1];
  long long dim = strtoll(argv[2], &pend, 10);
  int ite = atoi(argv[3]);
  string anonymous_prefix = argv[4];

  ExecutionContext ctx(anonymous_prefix, anonymous_prefix, anonymous_prefix, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);
  LOG_BEGIN();
  LOG_INFO("Loading data");
  GArray<float>* data = driver->readFromBinaryRecords<float>(training_path + ".data", dim*sizeof(float));
  GArray<int>* labels = driver->readFromBinaryRecords<int>(training_path + ".label", sizeof(int));
  LogisticRegressionModel lrm = LogisticRegressionModel(ctx, duration);
  lrm.train(data, labels, dim, 0.1, ite);
  float *w = lrm.getWeight();

  if(argc > 5){
    LOG_INFO("Test");
    string test_path = argv[5];
    GArray<float>* test_data = driver->readFromBinaryRecords<float>(test_path + ".data", dim*sizeof(float));
    GArray<int>* test_labels = driver->readFromBinaryRecords<int>(test_path + ".label", sizeof(int));
    LogisticRegressionPredictModel predict = LogisticRegressionPredictModel(ctx);
    predict.predict(test_data, w, dim, test_labels);
    float accu = predict.accuracy;
	if(rank == 0)
      cerr << "Test accuracy = " << accu*100.0 << "%" << endl;
  }

  MPI_Finalize();
  return 0;
}
