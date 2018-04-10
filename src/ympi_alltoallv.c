#include <assert.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <unistd.h>

#define assert(COND) do{if(!(COND)) {printf("ASSERTION VIOLATED, PROCESS pid = %d PAUSED\n", getpid()); while(1);}}while(0)

#define YMPI_VERBOSE

#ifdef YMPI_VERBOSE
#define YMPI_LINES  do{ MPI_Barrier(comm); if(rank == 0) { printf("    %s:%d\n", __FUNCTION__, __LINE__); } }while(0)
#else
#define YMPI_LINES
#endif

#define REGION_BEGIN() do{duration = -currentTimeUs();}while(0)
#define REGION_END(msg) do{duration += currentTimeUs(); if (rank == 0){printf("  [%s] %s Time: %lf sec\n", __FUNCTION__, msg, 1e-6 * duration);}}while(0)

#include <mpi.h>

#include "ympi.h"

// threshold

// set this to redirect all the YMPI to standard MPI implementation
// #define YMPI_USE_MPI

#ifdef YMPI_USE_MPI
#warning "[CAUTION] YMPI will use standard MPI implementation"
#endif

// maximum number of processes involving the communication [TODO: Confirm this number]
#define MAX_NUM_PROCS          (160*1024)

// maximum number of group size [TODO: Confirm this number]
// #define MAX_NUM_GROUPS         (256)
#define MAX_GROUP_SIZE         (256)

// performance-related parameter

// [CAUTION: Keep Three Constants Below Exactly (1), do not modify] for Mapreduce
#define NUM_TEMPORARY_BUFFER   (1)
#define NUM_RECEIVE_BUFFER     (1)
#define NUM_SEND_BUFFER        (1)

// for Alltoallv
#define A2A_NUM_TEMPORARY_BUFFER   (2)
#define A2A_NUM_RECEIVE_BUFFER     (2)
#define A2A_NUM_SEND_BUFFER        (2)
#define TEMPORARY_BUFFER_BYTES (1*1024*1024)

// utilities

#define TAG_MESG_1(DESTINATION)        (0x00000000 | (DESTINATION & 0xFFFFFF))
#define TAG_MESG_2(SOURCE)             (0x01000000 | (SOURCE & 0xFFFFFF))
#define TAG_MESG_1_LAST(DESTINATION)   (0x02000000 | (DESTINATION & 0xFFFFFF))
#define TAG_MESG_2_LAST(SOURCE)        (0x03000000 | (SOURCE & 0xFFFFFF))
#define TAG_MESG_1_INTACT(DESTINATION) (0x04000000 | (DESTINATION & 0xFFFFFF))
#define TAG_MESG_2_INTACT(SOURCE)      (0x05000000 | (SOURCE & 0xFFFFFF))
#define TAG_FIN                        (0x06000000)

// #define MPI_Test(...) do{printf("%d> %d AAA\n", rank, __LINE__); MPI_Test(__VA_ARGS__); printf("%d> %d BBB\n", rank, __LINE__);}while(0)

// void MPI_Comm_err_handler_function(MPI_Comm* comm, int* errcode) {
//   assert(0);
// }

static void sig_handler(int sig) {
  assert(0);
}

#define TAG_DESTINATION(TAG)     (TAG & 0xFFFFFF)
#define TAG_SOURCE(TAG)          (TAG & 0xFFFFFF)

// 0: mesg_1   1: mesg_2
#define TAG_PHASE(TAG)           ((TAG >> 24) & 1)

// 0: not last 1: last
//#define TAG_IS_LAST(TAG)         ((TAG >> 25) & 1) // TODO: Update with current scheme
#define TAG_IS_LAST(TAG)         (((TAG >> 25) & 1)||(TAG == TAG_FIN))
#define TAG_IS_INTACT(TAG)       ((TAG >> 26) & 1)

extern int g_group_size;

enum { MESG_STATE_AVAIL=0, MESG_STATE_RECV=1, MESG_STATE_SEND=2, MESG_STATE_SEND_DONE1=3 };

typedef struct YMPID_Temporary_buffer
{
  int         state;
  int         tag;      // should keep track on the tag when receiving
  size_t      bytes;
  MPI_Request req;      // recv request
  int         num_requests;
  MPI_Request reqs[MAX_GROUP_SIZE];
  char        temporary_buffer[TEMPORARY_BUFFER_BYTES];
} YMPID_Temporary_buffer;

typedef struct YMPID_Receive_buffer
{
  int         state;
  int         tag;
  size_t      bytes;
  MPI_Request req;
  char        recv_buffer[TEMPORARY_BUFFER_BYTES];
} YMPID_Receive_buffer;

typedef struct YMPID_Send_buffer
{
  int         state;
  MPI_Request req;
  int         next_proc;
} YMPID_Send_buffer;

// put an order to tbuffer 
static uint64_t               tbuffer_begin;
static uint64_t               tbuffer_end;
static YMPID_Temporary_buffer tbuffer[A2A_NUM_TEMPORARY_BUFFER];

static YMPID_Receive_buffer   rbuffer[A2A_NUM_RECEIVE_BUFFER];

static uint64_t               sbuffer_begin;
static uint64_t               sbuffer_end;
static YMPID_Send_buffer      sbuffer[A2A_NUM_SEND_BUFFER];

/* send in TEMPORARY_BUFFER_BYTES granuarity, thus we need to keep track on received bytes for each process */
static size_t                 received_bytes_pre[MAX_NUM_PROCS];
static size_t                 received_bytes_post[MAX_NUM_PROCS];
static size_t                 recvbytes[MAX_NUM_PROCS];
static size_t                 sendbytes[MAX_NUM_PROCS];

// column_sendbytes[g][o]: the number of bytes from g-th group which is relayed from me and destined to o-th offset
static size_t                 column_sendcounts[MAX_NUM_PROCS];
static size_t                 column_sendbytes[MAX_NUM_PROCS];

// static int deadlock;
// static double begin_time;
//#define WATCHDOG_INIT do{deadlock=0, begin_time=MPI_Wtime();}while(0)
//#define WATCHDOG_MONITOR do{double bt=MPI_Wtime(); if(bt-begin_time > 3 && !deadlock) {deadlock=1; LOGD("DEADLOCKED!\n");} }while(0)
#define WATCHDOG_INIT
#define WATCHDOG_MONITOR

int YMPI_AlltoallvL(const void *sendbuf, const size_t *sendcounts,
                         const size_t *sdispls, MPI_Datatype sendtype, void *recvbuf,
                         const size_t *recvcounts, const size_t *rdispls, MPI_Datatype recvtype,
                         MPI_Comm comm)
{
  WATCHDOG_INIT;
  
  int i, j;
  int rank, nprocs;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nprocs);
  assert(nprocs % g_group_size == 0);
  int num_group  = nprocs / g_group_size;
  int my_gid     = rank / g_group_size;
  int my_offset  = rank % g_group_size;

  YMPI_LINES;
  for (i=0; i<nprocs; i++) {
    assert(sendcounts[i] >= 0);
    assert(recvcounts[i] >= 0);
  }
  for (i=0; i<nprocs-1; i++) {
    assert(sdispls[i] >= 0);
    assert(sdispls[i] <= sdispls[i+1]);
    assert(rdispls[i] >= 0);
    assert(rdispls[i] <= rdispls[i+1]);
  }
  assert(sdispls[nprocs-1] >= 0);
  assert(rdispls[nprocs-1] >= 0);

  // MPI_Barrier(comm);
  // printf("  sendcounts of %d =", rank);
  // for(i=0; i<nprocs; i++) {
  //   printf(" %d", sendcounts[i]);
  // }
  // printf("\n");
  // printf("  recvcounts of %d =", rank);
  // for(i=0; i<nprocs; i++) {
  //   printf(" %d", recvcounts[i]);
  // }
  // printf("\n");
  // printf("  sdispls of %d =", rank);
  // for(i=0; i<nprocs; i++) {
  //   printf(" %d", sdispls[i]);
  // }
  // printf("\n");
  // printf("  rdispls of %d =", rank);
  // for(i=0; i<nprocs; i++) {
  //   printf(" %d", rdispls[i]);
  // }
  // printf("\n");
  // MPI_Barrier(comm);


  // assert(num_group < MAX_NUM_GROUPS);
  assert(g_group_size < MAX_GROUP_SIZE);

  MPI_Comm alltoallv_relay_comm;
  MPI_Comm alltoallv_column_comm;
  MPI_Comm relay_comm;
  MPI_Comm_dup(comm, &alltoallv_relay_comm);
  MPI_Comm_split(comm, my_offset, my_gid, &alltoallv_column_comm);
  relay_comm = alltoallv_relay_comm;

  // MPI_Errhandler errhandler;
  // MPI_Comm_create_errhandler(&MPI_Comm_err_handler_function,  &errhandler);
  // MPI_Comm_set_errhandler(comm, errhandler);
  // MPI_Comm_set_errhandler(alltoallv_column_comm, errhandler);
  // MPI_Comm_set_errhandler(alltoallv_relay_comm, errhandler);
  // struct sigaction act;
  // memset(&act, 0, sizeof(sigaction));
  // act.sa_handler = sig_handler;
  // sigaction(11, &act, NULL);

  {
    int ng;
    MPI_Comm_size(alltoallv_column_comm, &ng);
    assert(ng == num_group);
  }

  size_t extent;
  {
    MPI_Aint rlb, rextent, slb, sextent;
    MPI_Type_get_extent(recvtype, &rlb, &rextent);
    MPI_Type_get_extent(sendtype, &slb, &sextent);
    assert(rextent == sextent);
    extent = rextent;
  }

  MPI_Alltoall(sendcounts, g_group_size, MPI_UNSIGNED_LONG_LONG, column_sendcounts, g_group_size, MPI_UNSIGNED_LONG_LONG, alltoallv_column_comm);
  for(i=0; i<nprocs; i++) {
    column_sendbytes[i] = (size_t) column_sendcounts[i] * extent;
  }

  size_t temporary_bytes = TEMPORARY_BUFFER_BYTES / extent * extent;
  assert(temporary_bytes < (1LL<<31));

  /* termination condition:
      mesg1: num_group last MESG1
      mesg2: triggered by received_bytes_post reached recvcounts
  */
  int done_mesg_1    = 0;
  int done_mesg_2    = 0;
  for(i=0; i<nprocs; i++) {
    /*
    if(recvcounts[i] == 0) {
      done_mesg_2++;
    }
    */
    recvbytes[i] = (size_t) recvcounts[i] * extent;
    sendbytes[i] = (size_t) sendcounts[i] * extent;
  }
  int expect_mesg_1  = num_group;
  int expect_mesg_2  = 0;
  for(i=0; i<nprocs; i++) {
    if (recvbytes[i] > 0) {
      expect_mesg_2 ++;
    }
  }

  // LOGD("init done_mesg_2 = %d / %d\n", done_mesg_2, expect_mesg_2);

  // INITIALIZATION
  tbuffer_begin = 0;
  tbuffer_end   = 0;
  sbuffer_begin = 0;
  sbuffer_end   = 0;
  // memset(tbuffer, 0, sizeof(tbuffer));
  // memset(rbuffer, 0, sizeof(rbuffer));
  // memset(sbuffer, 0, sizeof(sbuffer));
  for(i=0; i<A2A_NUM_TEMPORARY_BUFFER; i++) {
    tbuffer[i].num_requests = 0;
    tbuffer[i].state = MESG_STATE_AVAIL;
  }
  for(i=0; i<A2A_NUM_RECEIVE_BUFFER; i++) {
    rbuffer[i].state = MESG_STATE_AVAIL;
  }
  for(i=0; i<A2A_NUM_SEND_BUFFER; i++) {
    sbuffer[i].state = MESG_STATE_AVAIL;
  }
  memset(received_bytes_pre, 0, nprocs * sizeof(size_t));
  memset(received_bytes_post, 0, nprocs * sizeof(size_t));

  // int num_relay_issued    = 0;
  // int num_relay_completed = 0;
  int num_sent_pre     = 0;
  int num_sent_post    = 0;
  int num_done1_pre    = 0;
  int num_done1_post   = 0;
  // int curr_send_dest   = (rank+1) % nprocs;
  int    curr_send_dest   = rank / g_group_size * g_group_size;
  size_t curr_send_offset = 0;

  YMPI_LINES;

  while (done_mesg_1 < expect_mesg_1 || done_mesg_2 < expect_mesg_2 || tbuffer_begin < tbuffer_end || num_sent_post < nprocs || num_done1_post < num_group) {
    WATCHDOG_MONITOR;
    int        flag;
    MPI_Status status;

    if (tbuffer_end - tbuffer_begin < A2A_NUM_TEMPORARY_BUFFER) {
      MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, relay_comm, &flag, &status);
      if (flag) {
        int source    = status.MPI_SOURCE;
        int tag       = status.MPI_TAG;
        int count;
        MPI_Get_count(&status, recvtype, &count);
        size_t rbytes = extent * count;
        int    phase  = TAG_PHASE(tag);
        assert(phase == 0);
        // mesg 1: forward via temporary buffer
        int    buf_id            = tbuffer_end % A2A_NUM_TEMPORARY_BUFFER;
        assert(tbuffer[buf_id].state == MESG_STATE_AVAIL);
        char* temporary_buffer   = tbuffer[buf_id].temporary_buffer;
        MPI_Request* req         = &tbuffer[buf_id].req;
        assert(rbytes <= temporary_bytes);
        assert(source >= 0);
        assert(source < nprocs);
        MPI_Irecv(temporary_buffer, rbytes, MPI_CHAR, source, tag, relay_comm, req);
        tbuffer[buf_id].state    = MESG_STATE_RECV;
        tbuffer[buf_id].tag      = tag;
        tbuffer[buf_id].bytes    = rbytes;
        // num_relay_issued++;
        tbuffer_end++;
      }
    }

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &flag, &status);
    if (flag) {
      int source    = status.MPI_SOURCE;
      int tag       = status.MPI_TAG;
      int count;
      MPI_Get_count(&status, recvtype, &count);
      size_t rbytes = extent * count;
      int phase     = TAG_PHASE(tag);
      assert(phase == 1);
      // mesg 2: recv into recv buffer
      int src   = TAG_SOURCE(tag); 
      for(i=0; i<A2A_NUM_RECEIVE_BUFFER; i++) {
        if(rbuffer[i].state == MESG_STATE_AVAIL) {
          int   buf_id             = i;
          size_t received_byte     = received_bytes_pre[src];
          received_bytes_pre[src]  += rbytes;
          char* rbuf               = ((char*) recvbuf) + extent*rdispls[src] + received_byte;
          MPI_Request* req         = &rbuffer[buf_id].req;
          // printf("%d> receive from rank %d (via %d) for %d bytes\n", rank, src, source, rbytes);
          MPI_Irecv(rbuf, rbytes, MPI_CHAR, source, tag, comm, req);
          rbuffer[buf_id].state    = MESG_STATE_RECV;
          break;
        }
      }
    }

    // progress & update pending tbuffer
    // for(i=0; i<A2A_NUM_TEMPORARY_BUFFER; i++) {
    if (tbuffer_end - tbuffer_begin > 0) {
      int i = tbuffer_begin % A2A_NUM_TEMPORARY_BUFFER;
      switch(tbuffer[i].state) {
        case MESG_STATE_AVAIL:
          assert(0);
          break;
        case MESG_STATE_RECV:
        {
          MPI_Status status;
          int        flag;
          MPI_Test(&tbuffer[i].req, &flag, &status);
          if(flag) {
            int source  = status.MPI_SOURCE;
            int tag     = status.MPI_TAG;
            int count;
            MPI_Get_count(&status, recvtype, &count);
            if(count != 0) {
              int is_intact = TAG_IS_INTACT(tag);
              if (is_intact) {
                int source_gid   = source / g_group_size;
                size_t buffer_bytes = tbuffer[i].bytes;
                int num_requests = 0;
                int curr_dest    = TAG_DESTINATION(tag);
                int curr_dest_offset = curr_dest % g_group_size;
                size_t sbytes      = 0;
                while (sbytes < buffer_bytes) {
                  assert(curr_dest_offset < g_group_size);
                  size_t curr_sendbytes = column_sendbytes[source_gid * g_group_size + curr_dest_offset];
                  if (curr_sendbytes > 0) {
                    assert(curr_sendbytes < (1LL<<31));
                    // printf("%d> relay from %d to %d for %d bytes\n", rank, source, curr_dest, curr_sendbytes);
                    MPI_Isend(&tbuffer[i].temporary_buffer[sbytes], (int) curr_sendbytes, MPI_CHAR, curr_dest, TAG_MESG_2(source), comm, &tbuffer[i].reqs[num_requests]);
                    sbytes += curr_sendbytes;
                    num_requests++;
                  }
                  // wrap around is not allowed because constraint
                  // TODO: on the way to reduce done_2 zero-sized message
                  curr_dest++;
                  curr_dest_offset++;
                }
                tbuffer[i].num_requests = num_requests;
                assert(sbytes == buffer_bytes);
                assert(curr_dest_offset <= g_group_size);
              } else {
                assert(tbuffer[i].bytes > 0);
                int dest    = TAG_DESTINATION(tag);
                MPI_Isend(tbuffer[i].temporary_buffer, tbuffer[i].bytes, MPI_CHAR, dest, TAG_MESG_2(source), comm, &tbuffer[i].reqs[0]);
                tbuffer[i].num_requests = 1;
                // LOGV("receive_mesg_1 send_from %d send_to %d\n", source, dest);
              }
              tbuffer[i].state = MESG_STATE_SEND;
            } else {
              // count == 0
              int is_last = TAG_IS_LAST(tag);
              assert(is_last);
              if(is_last) {
                done_mesg_1++;
                // LOGV("done_mesg_1 received from proc %d (%d/%d)\n", source, done_mesg_1, expect_mesg_1);
                // num_relay_completed++;
                // LOGV("num_relay_completed = (%d/%d)\n", num_relay_completed, num_relay_issued);
                tbuffer_begin++;
                assert(tbuffer_begin <= tbuffer_end);
                tbuffer[i].state = MESG_STATE_AVAIL;
              } else {
                // int is_intact = TAG_IS_INTACT(tag);
                // assert(!is_intact); // TODO: handle is_intact here
                // int dest    = TAG_DESTINATION(tag);
                // MPI_Isend(tbuffer[i].temporary_buffer, tbuffer[i].bytes, MPI_CHAR, dest, TAG_MESG_2(source), comm, &tbuffer[i].reqs[0]);
                // tbuffer[i].num_requests = 1;
                // // LOGV("receive_mesg_1 send_from %d send_to %d\n", source, dest);
                // tbuffer[i].state = MESG_STATE_SEND;
              }
            }
          }
          break;
        }
        case MESG_STATE_SEND:
        {
          MPI_Status status;
          int        flag;
          int        num_done = 0;
          int        num_requests = tbuffer[i].num_requests;
          for(j=0; j<num_requests; j++) {
            MPI_Test(&tbuffer[i].reqs[j], &flag, &status);
            if (flag) {
              num_done++;
            }
          }
          if (num_done == num_requests) {
            // num_relay_completed++;
            // LOGV("num_relay_completed = (%d/%d)\n", num_relay_completed, num_relay_issued);
            tbuffer_begin++;
            assert(tbuffer_begin <= tbuffer_end);
            tbuffer[i].state = MESG_STATE_AVAIL;
          }
          break;
        }
        default:
          printf("unknown tbuffer state\n");
          assert(0);
      }
    }

    // progress & update pending rbuffer
    for(i=0; i<A2A_NUM_RECEIVE_BUFFER; i++) {
      switch(rbuffer[i].state) {
        case MESG_STATE_AVAIL:
          break;
        case MESG_STATE_RECV:
        {
          MPI_Status status;
          int        flag;
          MPI_Test(&rbuffer[i].req, &flag, &status);
          if (flag) {
            int tag       = status.MPI_TAG;
            int source    = TAG_SOURCE(tag);
            int count;
            MPI_Get_count(&status, recvtype, &count);
            size_t rbytes = extent * count;
            received_bytes_post[source] += rbytes;
            // LOGV("receive %d bytes from %d (%d/%d)\n", rbytes, source, received_bytes_post[source], recvbytes[source]);
            if(received_bytes_post[source] == recvbytes[source]) {
              done_mesg_2++;
              // LOGV("done_mesg_2 from %d (%d/%d)\n", source, done_mesg_2, expect_mesg_2);
            }
            rbuffer[i].state = MESG_STATE_AVAIL;
          }
          break;
        }
        default:
          printf("unknown rbuffer state\n");
          assert(0);
      }
    }

    // try send
    if (sbuffer_end - sbuffer_begin < A2A_NUM_SEND_BUFFER) {
      int i = sbuffer_end % A2A_NUM_SEND_BUFFER;
      assert(sbuffer[i].state == MESG_STATE_AVAIL);
      int ok;
      if(num_sent_pre < nprocs) {
        int    gid    = curr_send_dest / g_group_size;
        // int    offset = curr_send_dest % g_group_size;
        // relayer dest
        int    dest   = gid * g_group_size + my_offset;
        if (curr_send_offset == 0 && sendbytes[curr_send_dest] <= temporary_bytes) {
          // intact mode
          size_t sbytes = 0;
          int    send_dest_begin = curr_send_dest;
          int    num_send_dests = 0;
          // all conditions must be satisfied
          //   1. total nprocs constraint
          //   2. local group constraint: must not cross group boundary
          //   3. curr_send_dest must not cross nprocs boundary in one transmission, (should)
          //   <could happen when num_groups=1>. This can be prevented by setting initial 
          //   curr_send_dest to the first member of the group, rather than (rank+1)%nprocs
          while (num_send_dests < (nprocs - num_sent_pre) && 
                 curr_send_dest / g_group_size == gid && 
                 sbytes + sendbytes[curr_send_dest] <= temporary_bytes) {
            sbytes += sendbytes[curr_send_dest];
            curr_send_dest = (curr_send_dest+1) % nprocs;
            num_send_dests++;
          }
          assert(num_send_dests > 0);
          // printf("%d> send to rank %d~%d (via %d) for %d bytes, num_sent_pre=%d, num_send_dests=%d\n", rank, send_dest_begin, send_dest_begin+num_send_dests-1, dest, sbytes, num_sent_pre, num_send_dests);
          if (sbytes > 0) {
            const char* buf = ((const char*) sendbuf) + extent*sdispls[send_dest_begin];
            assert(sbytes < (1LL<<31));
            ok = MPI_Isend(buf, (int) sbytes, MPI_CHAR, dest, TAG_MESG_1_INTACT(send_dest_begin), relay_comm, &sbuffer[i].req);
            assert(ok == MPI_SUCCESS);
          } else {
            sbuffer[i].req = MPI_REQUEST_NULL;
          }
          num_sent_pre += num_send_dests;
          assert(num_sent_pre <= nprocs);
          sbuffer[i].next_proc = num_send_dests;
          sbuffer[i].state = MESG_STATE_SEND;
        } else {
          // slice mode
          size_t sbytes = sendbytes[curr_send_dest] - curr_send_offset;
          if(sbytes > temporary_bytes) {
            sbytes = temporary_bytes;
          }
          const char* buf = ((const char*) sendbuf) + extent*sdispls[curr_send_dest] + curr_send_offset;
          curr_send_offset += sbytes;
          assert(sbytes > 0);
          // printf("%d> send to rank %d (via %d) for %d bytes, num_sent_pre=%d\n", rank, curr_send_dest, dest, sbytes, num_sent_pre);
          ok = MPI_Isend(buf, (int) sbytes, MPI_CHAR, dest, TAG_MESG_1(curr_send_dest), relay_comm, &sbuffer[i].req);
          assert(ok == MPI_SUCCESS);
          if(curr_send_offset == sendbytes[curr_send_dest]) {
            curr_send_offset = 0;
            curr_send_dest   = (curr_send_dest+1) % nprocs;
            num_sent_pre++;
            sbuffer[i].next_proc = 1;
          } else {
            sbuffer[i].next_proc = 0;
          }
          sbuffer[i].state = MESG_STATE_SEND;
        }
        sbuffer_end++;
      } else if (num_sent_post == nprocs && num_done1_pre < num_group) {
        int    dest   = num_done1_pre * g_group_size + my_offset;
        // printf("%d> done_1 sent to %d\n", rank, dest);
        ok = MPI_Isend(NULL, 0, MPI_CHAR, dest, TAG_FIN, relay_comm, &sbuffer[i].req);
        assert(ok == MPI_SUCCESS);
        num_done1_pre++;
        sbuffer[i].state = MESG_STATE_SEND_DONE1;
        sbuffer_end++;
      }
    }

    // progress send
    if (sbuffer_end > sbuffer_begin) {
      int i = sbuffer_begin % A2A_NUM_SEND_BUFFER;
      switch (sbuffer[i].state) {
        case MESG_STATE_SEND:
        {
          MPI_Status status;
          int        flag;
          // printf("%d> MPI_Test i = %d  state = %d   req = %x\n", rank, i, sbuffer[i].state, sbuffer[i].req);
          assert(i >= 0);
          assert(i < A2A_NUM_SEND_BUFFER);
          MPI_Test(&sbuffer[i].req, &flag, &status);
          if(flag) {
            num_sent_post += sbuffer[i].next_proc;
            assert(num_sent_post <= nprocs);
            // if(sbuffer[i].next_proc) {
            //   num_sent_post++;
            //   // LOGV("num_sent_post = (%d/%d)\n", num_sent_post, nprocs);
            // }
            sbuffer[i].state = MESG_STATE_AVAIL;
            sbuffer_begin++;
          }
          break;
        }
        case MESG_STATE_SEND_DONE1:
        {
          MPI_Status status;
          int        flag;
          MPI_Test(&sbuffer[i].req, &flag, &status);
          if(flag) {
            num_done1_post++;
            // printf("%d> num_done1_post++\n", rank);
            // LOGV("num_done1_post = (%d/%d)\n", num_done1_post, num_group);
            sbuffer[i].state = MESG_STATE_AVAIL;
            sbuffer_begin++;
          }
          break;
        }
        default:
          printf("unexpected sbuffer state\n");
          assert(0);
      }
    }
  }
  //printf("tbuffer_begin = %lu, tbuffer_end = %lu, eval = %d\n", tbuffer_begin, tbuffer_end, done_mesg_1 < expect_mesg_1 || done_mesg_2 < expect_mesg_2 || tbuffer_begin < tbuffer_end || num_sent_post < nprocs || num_done1_post < num_group);

  MPI_Comm_free(&alltoallv_relay_comm);
  MPI_Comm_free(&alltoallv_column_comm);

  YMPI_LINES;

  return MPI_SUCCESS;
}
