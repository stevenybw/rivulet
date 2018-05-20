#ifndef RIVULET_UTIL_H
#define RIVULET_UTIL_H

#include "common.h"

#include <immintrin.h>
#include <mpi.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/types.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <map>
#include <string>

using namespace std;

void  am_free(void* ptr);
void* am_memalign(size_t align, size_t size);
bool  is_power_of_2(uint64_t num);
void* malloc_pinned(size_t size);
void  memcpy_nts(void* dst, const void* src, size_t bytes);

void interleave_memory(void* ptr, size_t size, size_t chunk_size, int* node_list, int num_nodes);
void pin_memory(void* ptr, size_t size);
int  rivulet_numa_socket_bind(int socket_id);
void rivulet_yield();

extern std::mutex mu_mpi_routine;
using LockGuard = std::lock_guard<std::mutex>;

int MT_MPI_Cancel(MPI_Request *request);
int MT_MPI_Get_count(const MPI_Status *status, MPI_Datatype datatype, int *count);
int MT_MPI_Isend(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm, MPI_Request *request);
int MT_MPI_Irecv(void *buf, int count, MPI_Datatype datatype, int source, int tag, MPI_Comm comm, MPI_Request *request);
int MT_MPI_Test(MPI_Request *request, int *flag, MPI_Status *status);
int MT_MPI_Send(const void *buf, int count, MPI_Datatype datatype, int dest, int tag, MPI_Comm comm) ;

// Modular Approximation
template <int num_elem>
struct ApproxMod {
  uint8_t _table[num_elem];

  void init(int modular) {
    assert(modular < 256);
    // assert(num_elem / modular > 10);
    int _ne = num_elem;
    assert(_ne > 0);
    while((_ne & 1) == 0) {
      _ne >>= 1;
    }
    assert(_ne == 1);
    for (int i=0; i<num_elem; i++) {
      _table[i] = i % modular;
    }
  }

  // approximate num % modular
  uint8_t approx_mod(uint64_t num) {
    return _table[num & (num_elem-1)];
  }
};

using DefaultApproxMod = ApproxMod<128>;

struct ConfigFile 
{
  map<string, string> _config;
  ConfigFile(string config_path) {
    ifstream fin(config_path);
    if (!fin) {
      cerr << "Failed to open config file: " << config_path << endl;
      assert(false);
    }
    string key, equal, value;
    while (fin >> key >> equal >> value) {
      assert(equal == "=");
      _config[key] = value;
    }
  }
  uint64_t get_uint64(string key) {
    assert(_config.find(key) != _config.end());
    return stoul(_config[key]);
  }
};

// Configuration Parameters for Launching
struct LaunchConfig {
  int num_client_threads_per_socket;
  int num_updater_threads_per_socket;
  int num_export_threads_per_socket;
  int num_import_threads_per_socket;
  int num_sockets;
  int socket_list[8];
  int num_updater_sockets;
  int updater_socket_list[8];
  int num_comm_sockets;
  int comm_socket_list[8];

  void load_intra_socket(int num_client_threads, int num_updater_threads) {
    num_client_threads_per_socket = num_client_threads;
    num_updater_threads_per_socket = num_updater_threads;
    num_sockets = 1;
    socket_list[0] = 0;
    num_updater_sockets = 1;
    updater_socket_list[0] = 0;
  }

  void distributed_round_robin_socket(int rank, int num_sockets) {
    int socket_id       = rank % num_sockets;
    num_sockets         = 1;
    socket_list[0]      = socket_id;
    num_updater_sockets = 1;
    updater_socket_list[0] = socket_id;
    num_comm_sockets    = 1;
    comm_socket_list[0] = socket_id;
  }

  void load_from_config_file(std::string path) {
    std::string token1, token2;
    std::ifstream fin(path);
    assert(fin);
    
    fin >> token1 >> token2 >> num_client_threads_per_socket;
    assert(token1 == "num_client_threads_per_socket");
    assert(token2 == "=");

    fin >> token1 >> token2 >> num_updater_threads_per_socket;
    assert(token1 == "num_updater_threads_per_socket");
    assert(token2 == "=");

    fin >> token1 >> token2 >> num_export_threads_per_socket;
    assert(token1 == "num_export_threads_per_socket");
    assert(token2 == "=");

    fin >> token1 >> token2 >> num_import_threads_per_socket;
    assert(token1 == "num_import_threads_per_socket");
    assert(token2 == "=");

    fin >> token1 >> token2 >> num_sockets;
    assert(token1 == "num_sockets");
    assert(token2 == "=");

    fin >> token1 >> token2;
    assert(token1 == "socket_list");
    assert(token2 == "=");
    for(int i=0; i<num_sockets; i++) {
      fin >> socket_list[i];
    }

    fin >> token1 >> token2 >> num_updater_sockets;
    assert(token1 == "num_updater_sockets");
    assert(token2 == "=");

    fin >> token1 >> token2;
    assert(token1 == "updater_socket_list");
    assert(token2 == "=");
    for(int i=0; i<num_updater_sockets; i++) {
      fin >> updater_socket_list[i];
    }

    fin >> token1 >> token2 >> num_comm_sockets;
    assert(token1 == "num_comm_sockets");
    assert(token2 == "=");
    fin >> token1 >> token2;
    assert(token1 == "comm_socket_list");
    assert(token2 == "=");
    for(int i=0; i<num_comm_sockets; i++) {
      fin >> comm_socket_list[i];
    }
  }

  void _show_list(const char* name, int* list, int size) {
    cout << "  " << name << " = ";
    for (int i=0; i<size; i++) {
      cout << list[i] << " ";
    }
    cout << endl;
  }

  void show() {
    cout << "run config:" << endl;
    cout << "  num_client_threads_per_socket = " << num_client_threads_per_socket << endl;
    cout << "  num_updater_threads_per_socket = " << num_updater_threads_per_socket << endl;
    cout << "  num_export_threads_per_socket = " << num_export_threads_per_socket << endl;
    cout << "  num_import_threads_per_socket = " << num_import_threads_per_socket << endl;
    _show_list("socket_list", socket_list, num_sockets);
    _show_list("updater_socket_list", updater_socket_list, num_updater_sockets);
    _show_list("comm_socket_list", comm_socket_list, num_comm_sockets);
  }
};

struct AccumulateRequest {
  double* target;
  double rhs;
};

static constexpr int BATCH_SIZE = 32;
static thread_local size_t tl_num_reqs = 0;
static thread_local AccumulateRequest tl_acc_reqs[BATCH_SIZE];

template <typename T>
struct Atomic
{ };

template <>
struct Atomic<double> {
  const static double zero_value;
  
  static void update (double* p_lhs, double rhs) {
    (*p_lhs) += rhs;
  }

  static void cas_atomic_update(double* p_lhs, double rhs) {
    double lhs = *p_lhs;
    // uint64_t i_lhs = *((uint64_t*) &lhs);
    // dereferencing type-punned pointer will break strict-aliasing rules
    uint64_t i_lhs;
    memcpy(&i_lhs, &lhs, sizeof(uint64_t));
    while(true) {
      // total_attempt++;
      double new_lhs = lhs + rhs;
      uint64_t i_new_lhs;
      memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
      uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
      if (i_current_lhs == i_lhs) {
        break;
      }
      i_lhs = i_current_lhs;
      memcpy(&lhs, &i_lhs, sizeof(uint64_t));
      // num_retry++;
    }
  }

  static void rtm_atomic_update(double* p_lhs, double rhs) {
    // total_attempt++;
    if(_xbegin() == (unsigned int) -1) {
      *p_lhs += rhs;
      _xend();
    } else {
      _mm_pause();
      double lhs = *p_lhs;
      uint64_t i_lhs;
      memcpy(&i_lhs, &lhs, sizeof(uint64_t));
      while(true) {
        // total_attempt++;
        double new_lhs = lhs + rhs;
        uint64_t i_new_lhs;
        memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
        uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
        if (i_current_lhs == i_lhs) {
          break;
        }
        i_lhs = i_current_lhs;
        memcpy(&lhs, &i_lhs, sizeof(uint64_t));
        // num_retry++;
        _mm_pause();
      }
    }
  }

  static void batched_atomic_update_flush(const int num_reqs) {
    // total_attempt++;
    if(_xbegin() == (unsigned int) -1) {
      for(int i=0; i<num_reqs; i++) {
        *(tl_acc_reqs[i].target) += tl_acc_reqs[i].rhs;
      }
      _xend();
    } else {
      _mm_pause();
      for(int i=0; i<num_reqs; i++) {
        double* p_lhs = tl_acc_reqs[i].target;
        double rhs = tl_acc_reqs[i].rhs;

        double lhs = *p_lhs;
        uint64_t i_lhs;
        memcpy(&i_lhs, &lhs, sizeof(uint64_t));
        while(true) {
          // total_attempt++;
          double new_lhs = lhs + rhs;
          uint64_t i_new_lhs;
          memcpy(&i_new_lhs, &new_lhs, sizeof(uint64_t));
          uint64_t i_current_lhs = __sync_val_compare_and_swap((uint64_t*) p_lhs, i_lhs, i_new_lhs);
          if (i_current_lhs == i_lhs) {
            break;
          }
          i_lhs = i_current_lhs;
          memcpy(&lhs, &i_lhs, sizeof(uint64_t));
          // num_retry++;
          _mm_pause();
        }
      }
    }
    tl_num_reqs = 0;
  }

  static void batched_atomic_update(double* p_lhs, double rhs) {
    tl_acc_reqs[tl_num_reqs].target = p_lhs;
    tl_acc_reqs[tl_num_reqs].rhs = rhs;
    tl_num_reqs++;
    if (tl_num_reqs == BATCH_SIZE) {
      batched_atomic_update_flush(BATCH_SIZE);
    }
  }
};

template <typename Pred>
void wait_for(Pred pred) {
  //volatile uint64_t duration = 1024;
  while(!pred()) {
    //for(uint64_t i=0; i<duration; i++);
    //duration *= 2;
  }
}

#endif