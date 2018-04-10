#include <stdint.h>

int YMPI_Alltoallv(const void *sendbuf, const int *sendcounts,
                         const int *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const int *recvcounts, const int *rdispls, MPI_Datatype recvtype,
                         MPI_Comm comm);

int YMPI_AlltoallvL(const void *sendbuf, const size_t *sendcounts,
                         const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype,
                         MPI_Comm comm);

int YMPI_Alltoall(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                        void *recvbuf, int recvcount, MPI_Datatype recvtype,
                        MPI_Comm comm);

int YMPI_Alltoall_aggregate(const void *sendbuf, int sendcount, MPI_Datatype sendtype,
                        void *recvbuf, int recvcount, MPI_Datatype recvtype,
                        MPI_Comm comm);

int YMPI_Allgather(const void* sendbuf, int sendcount, MPI_Datatype sendtype, void *recvbuf, int recvcount, MPI_Datatype recvtype, MPI_Comm comm);

// YMPI_Sort: Caution
//   1. The content in 'base' will be corrupted!
//   2. The correct final result will be stored in 'intermediate'!
//   3. Please allocate 1x more space for 'intermediate' due to imbalance! It would be abundant
//  for sample-based pivoting method.
int YMPI_Sort(void* base,         // input array (corrupted after invocation)
              void* intermediate, // output array (about 1x more space is enough)
              size_t buf_bytes,   // number of bytes of 'intermediate' buffer
              size_t num_elements, // number of elements in 'base' input array
              size_t* r_num_elements, // number of resulting elements in 'intermediate' output array
              size_t element_width,   // number of bytes per element
              int (*compare)(const void* lhs, const void* rhs), // compare: the same as qsort
              MPI_Comm comm);     // comm: the communicator

int YMPI_Pagerank_order(const double* curr_val, const uint32_t* deg_out, uint64_t* global_order, uint32_t num_vertex, MPI_Comm comm);

#define LOGD(...) do {int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank); printf("[%4d]", rank); printf(__VA_ARGS__); }while(0)
#define LOGDS(...) do {int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);if (rank==0) {printf("[%4d]", rank); printf(__VA_ARGS__);}}while(0)

#define LOGV(...) do {int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank); printf("[%4d]", rank); printf(__VA_ARGS__); }while(0)
#define LOGVS(...) do {int rank;MPI_Comm_rank(MPI_COMM_WORLD, &rank);if (rank==0) {printf("[%4d]", rank); printf(__VA_ARGS__);}}while(0)

#define YMPI_GETGID(cpuid) ((cpuid)/g_group_size)
#define YMPI_GETOFF(cpuid) ((cpuid)%g_group_size)
#define YMPI_GETCPUID(gid, off) ((gid)*g_group_size + (off))