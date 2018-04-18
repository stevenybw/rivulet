#ifndef RIVULET_GRAPH_CONTEXT_H
#define RIVULET_GRAPH_CONTEXT_H

#include <mutex>
#include <thread>

#include <mpi.h>
#include <numa.h>
#include <numaif.h>
#include <omp.h>

#include "channel.h"
#include "common.h"
#include "graph.h"
#include "util.h"

template <typename NodeIdType>
struct VertexRange
{
  NodeIdType from;
  NodeIdType to;
  VertexRange(NodeIdType from, NodeIdType to) : from(from), to(to) {}
  struct Iterator {
    NodeIdType curr_vid;
    Iterator(NodeIdType curr_vid) : curr_vid(curr_vid) {}
    Iterator operator+(NodeIdType offset) { return Iterator(curr_vid + offset); }
    NodeIdType operator*() { return curr_vid; }
    NodeIdType operator-(Iterator rhs_it) { return curr_vid - rhs_it.curr_vid; }
  };
  Iterator begin() const { return Iterator(from); }
  Iterator end() const { return Iterator(to); }
};

struct GraphContext {
  template <typename T>
  struct __attribute__((packed)) GeneralUpdateRequest {
    uint32_t y;
    T contrib;
  };

  const static size_t CHANNEL_BYTES = 4096;
  const static size_t UPDATER_BLOCK_SIZE_PO2 = 4;
  const static size_t MPI_RECV_BUFFER_SIZE = 1*1024*1024;
  const static size_t MPI_SEND_BUFFER_SIZE = 1*1024*1024;
  const static int MAX_CLIENT_THREADS = 16;
  const static int MAX_UPDATER_THREADS = 16;
  const static int MAX_IMPORT_THREADS = 4;
  const static int MAX_EXPORT_THREADS = 4;
  const static int TAG_DATA = 100;
  const static int TAG_CLOSE = 101;

  using DefaultChannel = Channel_1<CHANNEL_BYTES>;

  std::mutex mu_mpi_routine;
  using LockGuard = std::lock_guard<std::mutex>;

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

  LaunchConfig config;
  DefaultApproxMod approx_mod; // for updater
  DefaultChannel channels[MAX_CLIENT_THREADS][MAX_UPDATER_THREADS];
  DefaultChannel to_exports[MAX_CLIENT_THREADS][MAX_EXPORT_THREADS];
  DefaultChannel from_imports[MAX_IMPORT_THREADS][MAX_UPDATER_THREADS];
  volatile int* num_client_done; // number of local client done
  volatile int* num_import_done; // number of local importer done
  volatile int* num_export_done; // number of local exporter done
  volatile int* current_chunk_id; // shared chunk_id for chunked work stealing
  volatile int* importer_num_close_request; // shared among importers, number of close requests received

  GraphContext(LaunchConfig config) : config(config) {
    assert(numa_available() != -1);
    numa_set_strict(1);

    int num_sockets = config.num_sockets;
    int num_updater_sockets = config.num_updater_sockets;
    int num_comm_sockets    = config.num_comm_sockets;
    int num_client_threads_per_socket = config.num_client_threads_per_socket;
    int num_updater_threads_per_socket = config.num_updater_threads_per_socket;
    int num_import_threads_per_socket = config.num_import_threads_per_socket;
    int num_export_threads_per_socket = config.num_export_threads_per_socket;
    int num_client_threads = num_client_threads_per_socket * num_sockets;
    int num_updater_threads = num_updater_threads_per_socket * num_updater_sockets;
    int num_import_threads  = num_import_threads_per_socket * num_comm_sockets;
    int num_export_threads  = num_export_threads_per_socket * num_comm_sockets;

    assert(num_client_threads  <= MAX_CLIENT_THREADS);
    assert(num_updater_threads <= MAX_UPDATER_THREADS);
    assert(num_import_threads  <= MAX_IMPORT_THREADS);
    assert(num_export_threads  <= MAX_EXPORT_THREADS);
    assert(is_power_of_2(num_import_threads));
    assert(is_power_of_2(num_export_threads));

    approx_mod.init(num_updater_threads);
    for(int i=0; i<num_client_threads; i++) {
      for(int j=0; j<num_updater_threads; j++) {
        channels[i][j].init();
      }
    }
    for(int i=0; i<num_client_threads; i++) {
      for(int j=0; j<num_export_threads; j++) {
        to_exports[i][j].init();
      }
    }
    for(int i=0; i<num_import_threads; i++) {
      for(int j=0; j<num_updater_threads; j++) {
        from_imports[i][j].init();
      }
    }
    num_client_done = new int;
    num_import_done = new int;
    num_export_done = new int;
    current_chunk_id = new int;
    importer_num_close_request = new int;
  }

  template <typename GraphType, typename VertexT, typename UpdateCallback>
  void edge_map(GraphType& graph, VertexT* curr_val, VertexT* next_val, uint32_t chunk_size, const UpdateCallback& update_op)
  {
    // using NodeT = typename GraphType::NodeType;
    // using IndexT = typename GraphType::IndexType;
    using UpdateRequest = GeneralUpdateRequest<VertexT>;

    int rank = graph._rank;
    int nprocs = graph._nprocs;

    *num_client_done = 0;
    *num_import_done = 0;
    *num_export_done = 0;
    *current_chunk_id = 0;
    *importer_num_close_request = 0;
    int num_sockets = config.num_sockets;
    int num_updater_sockets = config.num_updater_sockets;
    int num_comm_sockets = config.num_comm_sockets;
    int num_client_threads_per_socket = config.num_client_threads_per_socket;
    int num_updater_threads_per_socket = config.num_updater_threads_per_socket;
    int num_import_threads_per_socket = config.num_import_threads_per_socket;
    int num_export_threads_per_socket = config.num_export_threads_per_socket;
    int num_client_threads  = num_client_threads_per_socket * num_sockets;
    int num_updater_threads = num_updater_threads_per_socket * num_updater_sockets;
    int num_import_threads  = num_import_threads_per_socket * num_comm_sockets;
    int num_export_threads  = num_export_threads_per_socket * num_comm_sockets;
    int* socket_list = config.socket_list;
    int* updater_socket_list = config.updater_socket_list;
    int* comm_socket_list = config.comm_socket_list;

    assert(is_power_of_2(num_import_threads));
    assert(is_power_of_2(num_export_threads));

    if (rank == 0) {
      printf("EDGE_MAP BEGIN\n");
      printf("  graph_type = %s\n", GraphType::CLASS_NAME());
      printf("  total_num_nodes = %llu\n", graph.total_num_nodes());
      printf("  total_num_edges = %llu\n", graph.total_num_edges());
      printf("  channel_bytes = %zu\n", CHANNEL_BYTES);
      printf("  updater_block_size = %llu\n", (1LL<<UPDATER_BLOCK_SIZE_PO2));
    }

    if (num_client_threads < 1 || num_updater_threads < 1) {
      cout << "Insufficient number of threads: " << num_client_threads << ", " << num_updater_threads << endl;
      assert(false);
    }

    std::thread updater_threads[num_updater_threads];
    for (int i=0; i<num_updater_threads; i++) {
      updater_threads[i] = std::move(std::thread([this, &update_op](int id, int socket_id, int num_client_threads, int num_import_threads, volatile int* client_dones, volatile int* import_dones, VertexT* next_val) {
        uint64_t num_updates = 0;
        auto process_callback = [next_val, &update_op, &num_updates](const char* ptr, size_t bytes) {
          num_updates += bytes / sizeof(UpdateRequest);
          assert(bytes % sizeof(UpdateRequest) == 0);
          const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
          for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
            uint64_t llid = req_ptr->y;
            VertexT contrib = req_ptr->contrib;
            update_op(&next_val[llid], contrib);
            req_ptr++;
          }
        };
        int ok = numa_run_on_node(socket_id);
        assert (ok == 0);
        uint32_t counter = 0; // counter for round-robin
        while(!(*client_dones == num_client_threads && *import_dones == num_import_threads)) {
          uint32_t from = counter % num_client_threads;
          uint32_t import_id = counter & (num_import_threads-1);
          from_imports[import_id][id].poll(process_callback);
          channels[from][id].poll(process_callback);
          counter++;
          // for(int from=0; from<num_client_threads; from++) {
          //   channels[from][id].poll(process_callback);
          // }
          // for(int import_id=0; import_id<num_import_threads; import_id++) {
          //   from_imports[import_id][id].poll(process_callback);
          // }
        }
        printf(">>  updater thread %d request = %lu\n", id, num_updates);
      }, i, updater_socket_list[i/num_updater_threads_per_socket], num_client_threads, num_import_threads, num_client_done, num_import_done, next_val));
    }

    std::thread export_threads[num_export_threads];
    for (int i=0; i<num_export_threads; i++) {
      export_threads[i] = std::move(std::thread([this, rank, nprocs, &graph](int tid, int socket_id, int nprocs, int num_client_threads, volatile int* client_dones, volatile int* export_dones) {
        int ok = numa_run_on_node(socket_id);
        assert(ok == 0);
        int          buf_id_list[nprocs];
        size_t       curr_bytes_list[nprocs];
        MPI_Request  req[nprocs][2];
        char*        sendbuf[nprocs][2];
        for(int p=0; p<nprocs; p++) {
          buf_id_list[p] = 0;
          curr_bytes_list[p] = 0;
          for(int i=0; i<2; i++) {
            req[p][i] = MPI_REQUEST_NULL;
            sendbuf[p][i] = (char*) am_memalign(64, MPI_SEND_BUFFER_SIZE);
            assert(sendbuf[p][i]);
          }
        }
        while (*client_dones != num_client_threads) {
          for(int from=0; from<num_client_threads; from++) {
            to_exports[from][tid].poll([this, &graph, &buf_id_list, &curr_bytes_list, &req, &sendbuf](const char* ptr, size_t bytes) {
              assert(bytes % sizeof(UpdateRequest) == 0);
              const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
              for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
                uint64_t y = req_ptr->y;
                int next_val_rank = graph.get_rank_from_vid(y);
                uint32_t next_val_lid = graph.get_lid_from_vid(y);
                UpdateRequest update_request;
                update_request.y = next_val_lid;
                update_request.contrib = req_ptr->contrib;
                {
                  int    buf_id = buf_id_list[next_val_rank];
                  size_t curr_bytes = curr_bytes_list[next_val_rank];
                  if (curr_bytes + sizeof(UpdateRequest) > MPI_SEND_BUFFER_SIZE) {
                    // LINES;
                    int flag = 0;
                    // printf("  %d> send %zu bytes to %d\n", g_rank, curr_bytes, next_val_rank);
                    MT_MPI_Isend(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD, &req[next_val_rank][buf_id]);
                    while (!flag) {
                      MT_MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
                    }
                    // MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
                    buf_id = buf_id ^ 1;
                    curr_bytes = 0;
                    curr_bytes_list[next_val_rank] = curr_bytes;
                    buf_id_list[next_val_rank] = buf_id;
                  }
                  memcpy(&sendbuf[next_val_rank][buf_id][curr_bytes], &update_request, sizeof(UpdateRequest));
                  curr_bytes += sizeof(UpdateRequest);
                  curr_bytes_list[next_val_rank] = curr_bytes;
                }
                req_ptr++;
              }
            });
          }
        }
        for (int next_val_rank=0; next_val_rank<nprocs; next_val_rank++) {
          if (next_val_rank != rank) {
            int    buf_id = buf_id_list[next_val_rank];
            size_t curr_bytes = curr_bytes_list[next_val_rank];
            int flag = 0;
            while (!flag) {
              MT_MPI_Test(&req[next_val_rank][buf_id], &flag, MPI_STATUS_IGNORE);
            }
            req[next_val_rank][buf_id] = MPI_REQUEST_NULL;
            while (!flag) {
              MT_MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
            }
            req[next_val_rank][buf_id^1] = MPI_REQUEST_NULL;
            MT_MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
            MT_MPI_Send(NULL, 0, MPI_CHAR, next_val_rank, TAG_CLOSE, MPI_COMM_WORLD);
            buf_id = buf_id ^ 1;
            curr_bytes = 0;
            curr_bytes_list[next_val_rank] = curr_bytes;
            buf_id_list[next_val_rank] = buf_id;
          } else {
            assert(curr_bytes_list[rank] == 0); // no local update via export
          }
        }
        __sync_fetch_and_add(export_dones, 1);
        printf("[Export Thread] id %d done\n", tid);
      }, i, comm_socket_list[i/num_export_threads_per_socket], nprocs, num_client_threads, num_client_done, num_export_done));
    }

    std::thread import_threads[num_import_threads];
    for (int i=0; i<num_import_threads; i++) {
      import_threads[i] = std::move(std::thread([this, &graph](int tid, int socket_id, int nprocs, int num_export_threads, int num_updater_threads, volatile int* importer_num_close_request, volatile int* import_dones) {
        int ok = numa_run_on_node(socket_id);
        assert(ok == 0);
        int buf_id = 0;
        MPI_Request req[2];
        char* recvbuf[2];
        short counter[MAX_UPDATER_THREADS];
        memset(counter, 0, sizeof(counter));
        for(int i=0; i<2; i++) {
          recvbuf[i] = (char*) am_memalign(64, MPI_RECV_BUFFER_SIZE);
          assert(recvbuf[i]);
        }
        MT_MPI_Irecv(recvbuf[buf_id], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id]);
        int num_flush = 0;
        while (true) {
          int flag;
          MPI_Status st;
          MT_MPI_Test(&req[buf_id], &flag, &st);
          if (flag) {
            // printf("  %d> RECEIVED FROM %d\n", g_rank, st.MPI_SOURCE);
            MT_MPI_Irecv(recvbuf[buf_id^1], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id^1]);
            // int source = st.MPI_SOURCE;
            int tag    = st.MPI_TAG;
            int rbytes = 0;
            MT_MPI_Get_count(&st, MPI_CHAR, &rbytes);
            if (tag == TAG_DATA) {
              UpdateRequest* rbuf = (UpdateRequest*) recvbuf[buf_id];
              assert(rbytes % sizeof(UpdateRequest) == 0);
              int rcount = rbytes / sizeof(UpdateRequest);
              for(int i=0; i<rcount; i++) {
                uint32_t llid = rbuf[i].y;
                int target_tid = approx_mod.approx_mod(llid >> UPDATER_BLOCK_SIZE_PO2);
                size_t next_bytes = from_imports[tid][target_tid].push_explicit(rbuf[i], counter[target_tid]);
                counter[target_tid] = next_bytes;
              }
            } else if (tag == TAG_CLOSE) {
              assert(rbytes == 0);
              __sync_fetch_and_add(importer_num_close_request, 1);
              // printf("  %d> CLOSE RECEIVED (%d/%d)\n", g_rank, *importer_num_close_request, num_export_threads * nprocs);
            }
            buf_id = buf_id^1;
            // LINES;
          }
          if (*importer_num_close_request == num_export_threads * (nprocs-1)) {
            num_flush++;
            if (num_flush == 2) {
              break;
            }
          }
        }
        // printf("  %d> IMPORT EXIT\n", g_rank);
        MT_MPI_Cancel(&req[buf_id]);
        for (int i=0; i<2; i++) {
          req[i] = MPI_REQUEST_NULL;
          am_free(recvbuf[i]);
          recvbuf[i] = NULL;
        }
        for (int next_val=0; next_val<num_updater_threads; next_val++) {
          size_t curr_bytes = counter[next_val];
          from_imports[tid][next_val].flush_and_wait_explicit(curr_bytes);
          counter[next_val] = 0;
        }
        __sync_fetch_and_add(import_dones, 1);
      }, i, comm_socket_list[i/num_import_threads_per_socket], nprocs, num_export_threads, num_updater_threads, importer_num_close_request, num_import_done));
    }

    uint64_t edges_processed_per_thread[num_client_threads];
    uint64_t num_local_nodes = graph.local_num_nodes();
    uint32_t num_chunks      = num_local_nodes/chunk_size;
    *current_chunk_id        = 0;

    #pragma omp parallel num_threads(num_client_threads)
    {
      //register __m128i thread_local_state asm ("xmm15");
      //thread_local_state[0] = 0;
      short counter[MAX_UPDATER_THREADS];
      short export_counter[MAX_EXPORT_THREADS];
      memset(counter, 0, sizeof(counter));
      memset(export_counter, 0, sizeof(export_counter));

      uint64_t edges_processed;
      edges_processed = 0;
      int tid = omp_get_thread_num();
      int socket_id = socket_list[tid/num_client_threads_per_socket];
      int ok  = numa_run_on_node(socket_id);
      assert(ok == 0);
      while (true) {
        uint32_t chunk_id = __sync_fetch_and_add(current_chunk_id, 1);
        if (chunk_id >= num_chunks) {
          break;
        }
        uint32_t chunk_begin = chunk_id*chunk_size;
        uint32_t chunk_end;
        if (chunk_id == num_chunks - 1) {
          chunk_end = num_local_nodes;
        } else {
          chunk_end = (chunk_id+1)*chunk_size;
        }
        for (uint32_t i=chunk_begin; i<chunk_end; i++) {
          uint64_t from  = graph.get_index_from_lid(i);
          uint64_t to    = graph.get_index_from_lid(i+1);
          VertexT  contrib = curr_val[i];
          edges_processed += (to-from);
          for (uint64_t idx=from; idx<to; idx++) {
            uint32_t vid = graph.get_edge_from_index(idx);
            int target_rank = graph.get_rank_from_vid(vid);
            if (target_rank == rank) {
              uint32_t llid = graph.get_lid_from_vid(vid);
              UpdateRequest req;
              req.y=llid;
              req.contrib=contrib;
              int target_tid = approx_mod.approx_mod(llid >> UPDATER_BLOCK_SIZE_PO2);
              size_t next_bytes = channels[tid][target_tid].push_explicit(req, counter[target_tid]);
              counter[target_tid] = next_bytes;
            } else {
              UpdateRequest req;
              req.y = vid;
              req.contrib = contrib;
              int export_id = target_rank & (num_export_threads - 1); // equivalant to mode num_export_threads
              size_t next_bytes = to_exports[tid][export_id].push_explicit(req, export_counter[export_id]);
              export_counter[export_id] = next_bytes;
            }
          }
        }
      }
      for (int next_val=0; next_val<num_updater_threads; next_val++) {
        size_t curr_bytes = counter[next_val];
        channels[tid][next_val].flush_and_wait_explicit(curr_bytes);
        counter[next_val] = 0;
      }
      for (int export_id=0; export_id<num_export_threads; export_id++) {
        size_t curr_bytes = export_counter[export_id];
        to_exports[tid][export_id].flush_and_wait_explicit(curr_bytes);
        export_counter[export_id] = 0;
      }
      __sync_fetch_and_add(num_client_done, 1);
      edges_processed_per_thread[tid] = edges_processed;
    }

    for(int i=0; i<num_updater_threads; i++) {
      updater_threads[i].join();
    }
    for(int i=0; i<num_export_threads; i++) {
      export_threads[i].join();
    }
    for(int i=0; i<num_import_threads; i++) {
      import_threads[i].join();
    }

    uint64_t sum_ep = 0;
    uint64_t max_ep = 0;
    for(int i=0; i<num_client_threads; i++) {
      uint64_t ep = edges_processed_per_thread[i];
      printf("  %d> client thread %2d: processed edges = %lu\n", g_rank, i, ep);
      if (ep > max_ep) max_ep = ep;
      sum_ep += ep;
    }
    uint64_t global_sum_ep = 0;
    uint64_t global_max_ep = 0;
    MPI_Allreduce(&sum_ep, &global_sum_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&max_ep, &global_max_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    // printf("  avg_client_processed_edge = %lf\n", 1.0*sum_ep/num_client_threads);
    // printf("  max_client_processed_edge = %lf\n", 1.0*max_ep);
    if (rank == 0) {
      printf(">>> client load imbalance = %lf\n", (1.0*global_max_ep)/(1.0*global_sum_ep/nprocs/num_client_threads));
    }
  }

  template <typename ValueT, typename OnUpdateT, typename GraphType, typename VertexIterator, typename VertexValueCallback, typename OnUpdateGen>
  void compute_push_delegate(GraphType& graph, VertexIterator frontier_begin, VertexIterator frontier_end, VertexValueCallback vertex_value_op, OnUpdateGen& on_update_gen, uint32_t chunk_size)
  {
    // using NodeT = typename GraphType::NodeType;
    // using IndexT = typename GraphType::IndexType;
    using UpdateRequest = GeneralUpdateRequest<ValueT>;

    int rank = graph._rank;
    int nprocs = graph._nprocs;

    *num_client_done = 0;
    *num_import_done = 0;
    *num_export_done = 0;
    *current_chunk_id = 0;
    *importer_num_close_request = 0;
    int num_sockets = config.num_sockets;
    int num_updater_sockets = config.num_updater_sockets;
    int num_comm_sockets = config.num_comm_sockets;
    int num_client_threads_per_socket = config.num_client_threads_per_socket;
    int num_updater_threads_per_socket = config.num_updater_threads_per_socket;
    int num_import_threads_per_socket = config.num_import_threads_per_socket;
    int num_export_threads_per_socket = config.num_export_threads_per_socket;
    int num_client_threads  = num_client_threads_per_socket * num_sockets;
    int num_updater_threads = num_updater_threads_per_socket * num_updater_sockets;
    int num_import_threads  = num_import_threads_per_socket * num_comm_sockets;
    int num_export_threads  = num_export_threads_per_socket * num_comm_sockets;
    int* socket_list = config.socket_list;
    int* updater_socket_list = config.updater_socket_list;
    int* comm_socket_list = config.comm_socket_list;

    assert(is_power_of_2(num_import_threads));
    assert(is_power_of_2(num_export_threads));

    if (rank == 0) {
      printf("EDGE_MAP BEGIN\n");
      printf("  graph_type = %s\n", GraphType::CLASS_NAME());
      printf("  total_num_nodes = %llu\n", graph.total_num_nodes());
      printf("  total_num_edges = %llu\n", graph.total_num_edges());
      printf("  channel_bytes = %zu\n", CHANNEL_BYTES);
      printf("  updater_block_size = %llu\n", (1LL<<UPDATER_BLOCK_SIZE_PO2));
    }

    if (num_client_threads < 1 || num_updater_threads < 1) {
      cout << "Insufficient number of threads: " << num_client_threads << ", " << num_updater_threads << endl;
      assert(false);
    }

    std::thread updater_threads[num_updater_threads];
    for (int i=0; i<num_updater_threads; i++) {
      updater_threads[i] = std::move(std::thread([this, &on_update_gen](int id, int socket_id, int num_client_threads, int num_import_threads, volatile int* client_dones, volatile int* import_dones) {
        OnUpdateT on_update_op = on_update_gen();
        uint64_t num_updates = 0;
        auto process_callback = [&on_update_op, &num_updates](const char* ptr, size_t bytes) {
          num_updates += bytes / sizeof(UpdateRequest);
          assert(bytes % sizeof(UpdateRequest) == 0);
          const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
          for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
            uint64_t llid = req_ptr->y;
            ValueT contrib = req_ptr->contrib;
            on_update_op(llid, contrib);
            req_ptr++;
          }
        };
        int ok = numa_run_on_node(socket_id);
        assert (ok == 0);
        uint32_t counter = 0; // counter for round-robin
        while(!(*client_dones == num_client_threads && *import_dones == num_import_threads)) {
          uint32_t from = counter % num_client_threads;
          uint32_t import_id = counter & (num_import_threads-1);
          from_imports[import_id][id].poll(process_callback);
          channels[from][id].poll(process_callback);
          counter++;
          // for(int from=0; from<num_client_threads; from++) {
          //   channels[from][id].poll(process_callback);
          // }
          // for(int import_id=0; import_id<num_import_threads; import_id++) {
          //   from_imports[import_id][id].poll(process_callback);
          // }
        }
        printf(">>  updater thread %d request = %lu\n", id, num_updates);
      }, i, updater_socket_list[i/num_updater_threads_per_socket], num_client_threads, num_import_threads, num_client_done, num_import_done));
    }

    std::thread export_threads[num_export_threads];
    for (int i=0; i<num_export_threads; i++) {
      export_threads[i] = std::move(std::thread([this, rank, nprocs, &graph](int tid, int socket_id, int nprocs, int num_client_threads, volatile int* client_dones, volatile int* export_dones) {
        int ok = numa_run_on_node(socket_id);
        assert(ok == 0);
        int          buf_id_list[nprocs];
        size_t       curr_bytes_list[nprocs];
        MPI_Request  req[nprocs][2];
        char*        sendbuf[nprocs][2];
        for(int p=0; p<nprocs; p++) {
          buf_id_list[p] = 0;
          curr_bytes_list[p] = 0;
          for(int i=0; i<2; i++) {
            req[p][i] = MPI_REQUEST_NULL;
            sendbuf[p][i] = (char*) am_memalign(64, MPI_SEND_BUFFER_SIZE);
            assert(sendbuf[p][i]);
          }
        }
        while (*client_dones != num_client_threads) {
          for(int from=0; from<num_client_threads; from++) {
            to_exports[from][tid].poll([this, &graph, &buf_id_list, &curr_bytes_list, &req, &sendbuf](const char* ptr, size_t bytes) {
              assert(bytes % sizeof(UpdateRequest) == 0);
              const UpdateRequest* req_ptr = (const UpdateRequest*) ptr;
              for(size_t offset=0; offset<bytes; offset+=sizeof(UpdateRequest)) {
                uint64_t y = req_ptr->y;
                int next_val_rank = graph.get_rank_from_vid(y);
                uint32_t next_val_lid = graph.get_lid_from_vid(y);
                UpdateRequest update_request;
                update_request.y = next_val_lid;
                update_request.contrib = req_ptr->contrib;
                {
                  int    buf_id = buf_id_list[next_val_rank];
                  size_t curr_bytes = curr_bytes_list[next_val_rank];
                  if (curr_bytes + sizeof(UpdateRequest) > MPI_SEND_BUFFER_SIZE) {
                    // LINES;
                    int flag = 0;
                    // printf("  %d> send %zu bytes to %d\n", g_rank, curr_bytes, next_val_rank);
                    MT_MPI_Isend(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD, &req[next_val_rank][buf_id]);
                    while (!flag) {
                      MT_MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
                    }
                    // MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
                    buf_id = buf_id ^ 1;
                    curr_bytes = 0;
                    curr_bytes_list[next_val_rank] = curr_bytes;
                    buf_id_list[next_val_rank] = buf_id;
                  }
                  memcpy(&sendbuf[next_val_rank][buf_id][curr_bytes], &update_request, sizeof(UpdateRequest));
                  curr_bytes += sizeof(UpdateRequest);
                  curr_bytes_list[next_val_rank] = curr_bytes;
                }
                req_ptr++;
              }
            });
          }
        }
        for (int next_val_rank=0; next_val_rank<nprocs; next_val_rank++) {
          if (next_val_rank != rank) {
            int    buf_id = buf_id_list[next_val_rank];
            size_t curr_bytes = curr_bytes_list[next_val_rank];
            int flag = 0;
            while (!flag) {
              MT_MPI_Test(&req[next_val_rank][buf_id], &flag, MPI_STATUS_IGNORE);
            }
            req[next_val_rank][buf_id] = MPI_REQUEST_NULL;
            while (!flag) {
              MT_MPI_Test(&req[next_val_rank][buf_id^1], &flag, MPI_STATUS_IGNORE);
            }
            req[next_val_rank][buf_id^1] = MPI_REQUEST_NULL;
            MT_MPI_Send(sendbuf[next_val_rank][buf_id], curr_bytes, MPI_CHAR, next_val_rank, TAG_DATA, MPI_COMM_WORLD);
            MT_MPI_Send(NULL, 0, MPI_CHAR, next_val_rank, TAG_CLOSE, MPI_COMM_WORLD);
            buf_id = buf_id ^ 1;
            curr_bytes = 0;
            curr_bytes_list[next_val_rank] = curr_bytes;
            buf_id_list[next_val_rank] = buf_id;
          } else {
            assert(curr_bytes_list[rank] == 0); // no local update via export
          }
        }
        __sync_fetch_and_add(export_dones, 1);
        printf("[Export Thread] id %d done\n", tid);
      }, i, comm_socket_list[i/num_export_threads_per_socket], nprocs, num_client_threads, num_client_done, num_export_done));
    }

    std::thread import_threads[num_import_threads];
    for (int i=0; i<num_import_threads; i++) {
      import_threads[i] = std::move(std::thread([this, &graph](int tid, int socket_id, int nprocs, int num_export_threads, int num_updater_threads, volatile int* importer_num_close_request, volatile int* import_dones) {
        int ok = numa_run_on_node(socket_id);
        assert(ok == 0);
        int buf_id = 0;
        MPI_Request req[2];
        char* recvbuf[2];
        short counter[MAX_UPDATER_THREADS];
        memset(counter, 0, sizeof(counter));
        for(int i=0; i<2; i++) {
          recvbuf[i] = (char*) am_memalign(64, MPI_RECV_BUFFER_SIZE);
          assert(recvbuf[i]);
        }
        MT_MPI_Irecv(recvbuf[buf_id], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id]);
        int num_flush = 0;
        while (true) {
          int flag;
          MPI_Status st;
          MT_MPI_Test(&req[buf_id], &flag, &st);
          if (flag) {
            // printf("  %d> RECEIVED FROM %d\n", g_rank, st.MPI_SOURCE);
            MT_MPI_Irecv(recvbuf[buf_id^1], MPI_RECV_BUFFER_SIZE, MPI_CHAR, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req[buf_id^1]);
            // int source = st.MPI_SOURCE;
            int tag    = st.MPI_TAG;
            int rbytes = 0;
            MT_MPI_Get_count(&st, MPI_CHAR, &rbytes);
            if (tag == TAG_DATA) {
              UpdateRequest* rbuf = (UpdateRequest*) recvbuf[buf_id];
              assert(rbytes % sizeof(UpdateRequest) == 0);
              int rcount = rbytes / sizeof(UpdateRequest);
              for(int i=0; i<rcount; i++) {
                uint32_t llid = rbuf[i].y;
                int target_tid = approx_mod.approx_mod(llid >> UPDATER_BLOCK_SIZE_PO2);
                size_t next_bytes = from_imports[tid][target_tid].push_explicit(rbuf[i], counter[target_tid]);
                counter[target_tid] = next_bytes;
              }
            } else if (tag == TAG_CLOSE) {
              assert(rbytes == 0);
              __sync_fetch_and_add(importer_num_close_request, 1);
              // printf("  %d> CLOSE RECEIVED (%d/%d)\n", g_rank, *importer_num_close_request, num_export_threads * nprocs);
            }
            buf_id = buf_id^1;
            // LINES;
          }
          if (*importer_num_close_request == num_export_threads * (nprocs-1)) {
            num_flush++;
            if (num_flush == 2) {
              break;
            }
          }
        }
        // printf("  %d> IMPORT EXIT\n", g_rank);
        MT_MPI_Cancel(&req[buf_id]);
        for (int i=0; i<2; i++) {
          req[i] = MPI_REQUEST_NULL;
          am_free(recvbuf[i]);
          recvbuf[i] = NULL;
        }
        for (int i=0; i<num_updater_threads; i++) {
          size_t curr_bytes = counter[i];
          from_imports[tid][i].flush_and_wait_explicit(curr_bytes);
          counter[i] = 0;
        }
        __sync_fetch_and_add(import_dones, 1);
      }, i, comm_socket_list[i/num_import_threads_per_socket], nprocs, num_export_threads, num_updater_threads, importer_num_close_request, num_import_done));
    }

    uint64_t edges_processed_per_thread[num_client_threads];
    uint64_t frontier_num_nodes = frontier_end - frontier_begin;
    uint32_t num_chunks         = frontier_num_nodes/chunk_size;
    // uint64_t num_local_nodes = graph.local_num_nodes();
    *current_chunk_id           = 0;

    #pragma omp parallel num_threads(num_client_threads)
    {
      //register __m128i thread_local_state asm ("xmm15");
      //thread_local_state[0] = 0;
      short counter[MAX_UPDATER_THREADS];
      short export_counter[MAX_EXPORT_THREADS];
      memset(counter, 0, sizeof(counter));
      memset(export_counter, 0, sizeof(export_counter));

      uint64_t edges_processed;
      edges_processed = 0;
      int tid = omp_get_thread_num();
      int socket_id = socket_list[tid/num_client_threads_per_socket];
      int ok  = numa_run_on_node(socket_id);
      assert(ok == 0);
      while (true) {
        uint32_t chunk_id = __sync_fetch_and_add(current_chunk_id, 1);
        if (chunk_id >= num_chunks) {
          break;
        }
        uint32_t chunk_begin = chunk_id*chunk_size;
        uint32_t chunk_end;
        if (chunk_id == num_chunks - 1) {
          chunk_end = frontier_num_nodes;
        } else {
          chunk_end = (chunk_id+1)*chunk_size;
        }
        for (uint32_t i=chunk_begin; i<chunk_end; i++) {
          uint64_t u     = *(frontier_begin + i);
          uint64_t from  = graph.get_index_from_lid(u);
          uint64_t to    = graph.get_index_from_lid(u+1);
          ValueT  contrib = vertex_value_op(u);
          edges_processed += (to-from);
          for (uint64_t idx=from; idx<to; idx++) {
            uint32_t vid = graph.get_edge_from_index(idx);
            int target_rank = graph.get_rank_from_vid(vid);
            if (target_rank == rank) {
              uint32_t llid = graph.get_lid_from_vid(vid);
              UpdateRequest req;
              req.y=llid;
              req.contrib=contrib;
              int target_tid = approx_mod.approx_mod(llid >> UPDATER_BLOCK_SIZE_PO2);
              size_t next_bytes = channels[tid][target_tid].push_explicit(req, counter[target_tid]);
              counter[target_tid] = next_bytes;
            } else {
              UpdateRequest req;
              req.y = vid;
              req.contrib = contrib;
              int export_id = target_rank & (num_export_threads - 1); // equivalant to mode num_export_threads
              size_t next_bytes = to_exports[tid][export_id].push_explicit(req, export_counter[export_id]);
              export_counter[export_id] = next_bytes;
            }
          }
        }
      }
      for (int next_val=0; next_val<num_updater_threads; next_val++) {
        size_t curr_bytes = counter[next_val];
        channels[tid][next_val].flush_and_wait_explicit(curr_bytes);
        counter[next_val] = 0;
      }
      for (int export_id=0; export_id<num_export_threads; export_id++) {
        size_t curr_bytes = export_counter[export_id];
        to_exports[tid][export_id].flush_and_wait_explicit(curr_bytes);
        export_counter[export_id] = 0;
      }
      __sync_fetch_and_add(num_client_done, 1);
      edges_processed_per_thread[tid] = edges_processed;
    }

    for(int i=0; i<num_updater_threads; i++) {
      updater_threads[i].join();
    }
    for(int i=0; i<num_export_threads; i++) {
      export_threads[i].join();
    }
    for(int i=0; i<num_import_threads; i++) {
      import_threads[i].join();
    }

    uint64_t sum_ep = 0;
    uint64_t max_ep = 0;
    for(int i=0; i<num_client_threads; i++) {
      uint64_t ep = edges_processed_per_thread[i];
      printf("  %d> client thread %2d: processed edges = %lu\n", g_rank, i, ep);
      if (ep > max_ep) max_ep = ep;
      sum_ep += ep;
    }
    uint64_t global_sum_ep = 0;
    uint64_t global_max_ep = 0;
    MPI_Allreduce(&sum_ep, &global_sum_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    MPI_Allreduce(&max_ep, &global_max_ep, 1, MPI_UNSIGNED_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    // printf("  avg_client_processed_edge = %lf\n", 1.0*sum_ep/num_client_threads);
    // printf("  max_client_processed_edge = %lf\n", 1.0*max_ep);
    if (rank == 0) {
      printf(">>> client load imbalance = %lf\n", (1.0*global_max_ep)/(1.0*global_sum_ep/nprocs/num_client_threads));
    }
  }

  template <typename NodeT, typename IndexT, typename VertexT, typename UpdateCallback>
  void edge_map_smp_pull(SharedGraph<NodeT, IndexT>& graph_t, VertexT* curr_val, VertexT* next_val, uint32_t chunk_size, const UpdateCallback& update_op) {
    assert(graph_t.nprocs() == 1);
    size_t num_nodes = graph_t.total_num_nodes();
    #pragma omp parallel
    {
      uint32_t num_parts = num_nodes/chunk_size;
      #pragma omp for schedule(dynamic, 1)
      for (uint32_t part_id=0; part_id<num_parts; part_id++) {
        uint32_t chunk_begin = part_id * chunk_size;
        uint32_t chunk_end;
        if (part_id == num_parts - 1) {
          chunk_end = num_nodes;
        } else {
          chunk_end = (part_id+1)*chunk_size;
        }
        for (uint32_t y=chunk_begin; y<chunk_end; y++) {
          uint64_t from  = graph_t._index[y];
          uint64_t to    = graph_t._index[y+1];
          for (uint64_t idx=from; idx<to; idx++) {
            uint32_t x = graph_t._edges[idx];
            double contrib = curr_val[x];
            next_val[y] += contrib;
          }
        }
      }
    }
  }

  template <typename NodeT, typename IndexT, typename VertexT, typename ProtectedUpdateCallback>
  void edge_map_smp_push(SharedGraph<NodeT, IndexT>& graph, VertexT* curr_val, VertexT* next_val, uint32_t chunk_size, const ProtectedUpdateCallback& update_op) {
    assert(graph.nprocs() == 1);
    size_t num_nodes = graph.total_num_nodes();
    #pragma omp parallel
    {
      uint32_t num_parts = num_nodes/chunk_size;
      #pragma omp for schedule(dynamic, 1)
      for (uint32_t part_id=0; part_id<num_parts; part_id++) {
        uint32_t chunk_begin = part_id*chunk_size;
        uint32_t chunk_end;
        if (part_id == num_parts - 1) {
          chunk_end = num_nodes;
        } else {
          chunk_end = (part_id+1)*chunk_size;
        }
        for(uint32_t i=chunk_begin; i<chunk_end; i++) {
          uint64_t from = graph._index[i];
          uint64_t to   = graph._index[i+1];
          VertexT  contrib = curr_val[i];
          for (uint64_t idx=from; idx<to; idx++) {
            uint32_t y = graph._edges[idx];
            update_op(&next_val[y], contrib);
          }
        }
      }
    }
  }
};


#endif