#include "stream_buffer.h"

#include <algorithm>
#include <functional>
#include <thread>

#include <mpi.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

const int TAG_DATA = 100;

#define VALIDATION 0

template <size_t num_buffers = 2>
class MPIStreamBufferHandlerGeneral : public StreamBufferHandler
{
public:
  MPIStreamBufferHandlerGeneral() : comm(MPI_COMM_NULL), target_rank(-1), rank(-1) {
    for(size_t i=0; i<num_buffers; i++) {
      occupied[i] = false;
      reqs[i] = MPI_REQUEST_NULL;
    }
  }

  MPIStreamBufferHandlerGeneral(MPI_Comm comm, int target_rank) : comm(comm), target_rank(target_rank) {
    MPI_Comm_rank(comm, &rank);
    for(size_t i=0; i<num_buffers; i++) {
      occupied[i] = false;
      reqs[i] = MPI_REQUEST_NULL;
    }
  }

  void on_issue(int buffer_id, char* buffer, size_t bytes) override {
    PRINTF("%d> on_issue  id=%d/%d   bytes=%zu\n", rank, target_rank, buffer_id, bytes);
    assert(!occupied[buffer_id]);
    int bytes_as_i32 = bytes;
    PRINTF("%d> send to rank %d for %zu bytes\n", rank, target_rank, bytes);
    MPI_Isend(buffer, bytes_as_i32, MPI_CHAR, target_rank, TAG_DATA, comm, &reqs[buffer_id]);
    occupied[buffer_id] = true;
  }

  void on_wait(int buffer_id) override {
    PRINTF("%d> on_wait  id=%d/%d\n", rank, target_rank, buffer_id);
    assert(occupied[buffer_id]);
    MPI_Wait(&reqs[buffer_id], MPI_STATUS_IGNORE);
    occupied[buffer_id] = false;
  }

private:
  MPI_Comm comm;
  int target_rank;
  int rank;
  bool occupied[num_buffers];
  MPI_Request reqs[num_buffers];
};

using MPIStreamBufferHandler = MPIStreamBufferHandlerGeneral<2>;
using MPIStreamBuffer = StreamBuffer<MPIStreamBufferHandler, 2, 4096>;

/*! \brief Worker
 *
 *  Send num_element worker_id to each process
 */
void sender(int worker_id, int nprocs, pthread_barrier_t* barrier, MPIStreamBuffer* sbs, size_t num_element)
{
  MPIStreamBuffer::ThreadContext ctx[nprocs];
  for (size_t i=0; i<num_element; i++) {
    for (int dst=0; dst<nprocs; dst++) {
      uint64_t message = worker_id;
      sbs[dst].push(ctx[dst], message);
    }
  }
  for(int dst=0; dst<nprocs; dst++) {
    sbs[dst].flush(ctx[dst], barrier);
  }
}

void receiver(MPI_Comm comm, int nprocs, int num_threads, size_t num_element, int worker_id, size_t mpi_buffer_capacity)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  const size_t num_recv_buffer = 2;
  const size_t max_num_threads = 256;
  int curr_buffer_id = 0;
  MPI_Request reqs[num_recv_buffer];
  char* recv_buffer[num_recv_buffer];
  for(int i=0; i<num_recv_buffer; i++) {
    recv_buffer[i] = (char*) memalign(4096, mpi_buffer_capacity);
    MPI_Irecv(recv_buffer[i], mpi_buffer_capacity, MPI_CHAR, MPI_ANY_SOURCE, TAG_DATA, comm, &reqs[i]);
  }
  size_t worker_id_count[max_num_threads];
  for(size_t i=0; i<max_num_threads; i++) {
    worker_id_count[i] = 0;
  }
  while (true) {
    MPI_Status st;
    MPI_Wait(&reqs[curr_buffer_id], &st);
    int count;
    MPI_Get_count(&st, MPI_CHAR, &count);
    if (count == 0) {
      break;
    }
    uint64_t* messages = (uint64_t*) recv_buffer[curr_buffer_id];
    assert(count % sizeof(uint64_t) == 0);
    size_t num_element = count / sizeof(uint64_t);
    PRINTF("%d> receive from rank %d for %zu elements\n", rank, st.MPI_SOURCE, num_element);
#if VALIDATION
    size_t i = 0;
    if (num_element >= 4) {
      for(; i<num_element-4; i+=4) {
        uint64_t message0 = messages[i];
        uint64_t message1 = messages[i+1];
        uint64_t message2 = messages[i+2];
        uint64_t message3 = messages[i+3];
        assert(message0 < max_num_threads);
        assert(message1 < max_num_threads);
        assert(message2 < max_num_threads);
        assert(message3 < max_num_threads);
        worker_id_count[message0]++;
        worker_id_count[message1]++;
        worker_id_count[message2]++;
        worker_id_count[message3]++;
      }
    }
    for(; i<num_element; i++) {
      uint64_t message = messages[i];
      assert(message < max_num_threads);
      worker_id_count[message]++;
    }
#endif
    MPI_Irecv(recv_buffer[curr_buffer_id], mpi_buffer_capacity, MPI_CHAR, MPI_ANY_SOURCE, TAG_DATA, comm, &reqs[curr_buffer_id]);
    curr_buffer_id^=1;
  }
#if VALIDATION
  for (size_t i=0; i<max_num_threads; i++) {
    assert(worker_id_count[i] == 0 || worker_id_count[i] == (nprocs * num_element));
  }
#endif
  for(int i=0; i<num_recv_buffer; i++) {
    free(recv_buffer[i]);
  }
}

int main(int argc, char* argv[])
{
  int required_level = MPI_THREAD_MULTIPLE;
  int provided_level;
  MPI_Init_thread(&argc, &argv, required_level, &provided_level);
  assert(provided_level >= required_level);
  init_debug();
  int rank, nprocs;
  MPI_Comm barrier_comm;
  MPI_Comm_dup(MPI_COMM_WORLD, &barrier_comm);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

  g_rank = rank;
  g_nprocs = nprocs;

  CommandLine commandLine(argc, argv, 3, {"num_send_threads", "num_recv_threads", "num_element_per_thread"}, {"1", "1", "65536"});

  int num_threads = atoi(commandLine.getValue(0));
  int num_recv_threads = atoi(commandLine.getValue(1));
  size_t num_element_per_thread = atoll(commandLine.getValue(2));

  printf("num_sender_threads = %d\n", num_threads);
  printf("num_recv_threads = %d\n", num_recv_threads);
  printf("num_element_per_thread = %zu\n", num_element_per_thread);

  size_t mpi_buffer_capacity = 1LL<<20; // 1 MB

  MPIStreamBuffer sbs[nprocs];
  for(int dst=0; dst<nprocs; dst++) {
    sbs[dst] = std::move(MPIStreamBuffer(mpi_buffer_capacity, MPIStreamBufferHandler(MPI_COMM_WORLD, dst)));
  }

  pthread_barrier_t barrier;
  pthread_barrier_init(&barrier, NULL, num_threads);
  std::thread sender_threads[num_threads];
  std::thread recver_threads[num_recv_threads];

  uint64_t duration = -currentTimeUs();
  for(int i=0; i<num_threads; i++) {
    sender_threads[i] = std::move(std::thread(sender, i, nprocs, &barrier, &sbs[0], num_element_per_thread));
  }
  for(int i=0; i<num_recv_threads; i++) {
    recver_threads[i] = std::move(std::thread(receiver, MPI_COMM_WORLD, nprocs, num_threads, num_element_per_thread, i, mpi_buffer_capacity));
  }
  for(int i=0; i<num_threads; i++) {
    sender_threads[i].join();
  }
  MPI_Barrier(barrier_comm);
  printf("All sender complete, terminate receiver\n");
  for (int i=0; i<num_recv_threads; i++) {
    MPI_Send(NULL, 0, MPI_CHAR, rank, TAG_DATA, MPI_COMM_WORLD);
  }
  for(int i=0; i<num_recv_threads; i++) {
    recver_threads[i].join();
  }
  MPI_Barrier(barrier_comm);
  duration += currentTimeUs();
  size_t bytes = 1LL*num_threads*num_element_per_thread*nprocs;
  printf("Done, emitted = %lf GB, node throughput = %lf MB/s\n", (1e-9*bytes), (1.0 * bytes / duration));

  MPI_Finalize();

  return 0;
}