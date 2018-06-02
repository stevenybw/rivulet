#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <numa.h>
#include <numaif.h>
#include <sched.h>
#include <time.h>

#include <mpi.h>
#include <signal.h>

#include "util.h"

void* pages[128*1024];
int nodes[128*1024];
int statuses[128*1024];

uint64_t g_begin_ts;
int g_rank = -1;
int g_nprocs = -1;

#ifdef MPI_DEBUG

static void MPI_Comm_err_handler_function(MPI_Comm* comm, int* errcode, ...) {
  assert(0);
}

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

void am_free(void* ptr) {
  free(ptr);
}

void* am_memalign(size_t align, size_t size) {
  void* ptr;
  int ok = posix_memalign(&ptr, align, size);
  assert(ok == 0);
  return ptr;
}

uint64_t currentAbsoluteTimestamp() {
  unsigned hi, lo;
  asm volatile ("CPUID\n\t"
      "RDTSC\n\t"
      "mov %%edx, %0\n\t"
      "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo) : : "%rax", "%rbx", "%rcx", "%rdx");
  return ((uint64_t) hi << 32) | lo;
}

uint64_t currentTimestamp() {
  return currentAbsoluteTimestamp() - g_begin_ts;
}

uint64_t currentTimeUs()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_usec + 1000000 * tv.tv_sec;
}

char* currentDatetime()
{
  time_t t = time(nullptr);
  return asctime(localtime(&t));
}

string filename_append_postfix(string filename, int rank, int nprocs) {
  return filename + "." + to_string(rank) + "." + to_string(nprocs);
}

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

void* malloc_pinned(size_t size) {
  // !NOTE: use memalign to allocate 256-bytes aligned buffer
  void* ptr = am_memalign(4096, size);
  // if (mlock(ptr, size)<0) {
  //   perror("mlock failed");
  //   assert(false);
  // }
  return ptr;
}

// void memcpy_nts(void* dst, const void* src, size_t bytes) {
//   if (bytes == 16) {
//     _mm_stream_pd((double*)dst, *((const __m128d*)src));
//   } else {
//     __m256i* dst_p = (__m256i*) dst;
//     const __m256i* src_p = (const __m256i*) src;
//     size_t off = 0;
//     // _mm_stream_ps((float*) &(_window->data[seq][curr_bytes]), *((__m128*) &data));
//     for(; off<bytes; off+=32) {
//       _mm256_stream_si256(dst_p, *src_p);
//       dst_p++;
//       src_p++;
//     }
//   }
// }

inline static void* align_left(void* ptr, size_t index) {
  uintptr_t ptr_val = (uintptr_t) ptr;
  ptr_val = ((ptr_val >> index) << index);
  return (void*)ptr_val;
}

inline static void* align_right(void* ptr, size_t index) {
  uintptr_t ptr_val = (uintptr_t) ptr;
  size_t mask = (1LL << index) - 1;
  ptr_val = ((ptr_val+mask)&(~mask));
  return (void*)ptr_val;
}

void memcpy_nts(void* dst, const void* src, size_t bytes) {
  
}

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

void pin_memory(void* ptr, size_t size) {
  if (mlock(ptr, size) < 0) {
    perror("mlock failed");
    assert(false);
  }
}

int  rivulet_numa_socket_bind(int socket_id) {
  //return numa_run_on_node(socket_id);
  return 0;
}

void rivulet_yield() {
  sched_yield();
}

namespace memory {
  void* allocate_shared_rw(size_t bytes) {
    void* ptr = NULL;
    int ok = posix_memalign(&ptr, 64, bytes);
    assert(ok == 0);
    assert(ptr != NULL);
    return ptr;
  }

  void free(void* ptr) {
    ::free(ptr);
  }

  void* numa_alloc_onnode(size_t bytes, int node_id) {
    return ::numa_alloc_onnode(bytes, node_id);
  }
  void  numa_free(void* start, size_t size) {
    ::numa_free(start, size);
  }
};

std::mutex mu_mpi_routine;

void RV_Init()
{
  #ifdef MULTITHREAD
  int required_level = MPI_THREAD_MULTIPLE;
  #else
  int required_level = MPI_THREAD_SERIALIZED;
  #endif

  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  assert(provided_level >= required_level);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
  g_begin_ts = currentAbsoluteTimestamp();
}

void RV_Finalize()
{
  MPI_Finalize();
}

#ifdef MULTITHREAD

int MT_MPI_Cancel(MPI_Request *request)
{
  return MPI_Cancel(request);
}

int MT_MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
{
  return MPI_Get_count(status, datatype, count);
}

int MT_MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MT_MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int MT_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
  return MPI_Test(request, flag, status);
}

int MT_MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) 
{
  return MPI_Send(buf, count, datatype, dest, tag, comm);
}

#else

int MT_MPI_Cancel(MPI_Request *request)
{
  LockGuard lk(mu_mpi_routine);
  return MPI_Cancel(request);
}

int MT_MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count)
{
  LockGuard lk(mu_mpi_routine);
  return MPI_Get_count(status, datatype, count);
}

int MT_MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request)
{
  LockGuard lk(mu_mpi_routine);
  return MPI_Isend(buf, count, datatype, dest, tag, comm, request);
}

int MT_MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request)
{
  LockGuard lk(mu_mpi_routine);
  return MPI_Irecv(buf, count, datatype, source, tag, comm, request);
}

int MT_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status)
{
  LockGuard lk(mu_mpi_routine);
  return MPI_Test(request, flag, status);
}

int MT_MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) 
{
  MPI_Request req;
  MT_MPI_Isend(buf, count, datatype, dest, tag, comm, &req);
  int flag = 0;
  while (!flag) {
    MT_MPI_Test(&req, &flag, MPI_STATUS_IGNORE);
  }
  return MPI_SUCCESS;
}

#endif