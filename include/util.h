#ifndef RIVULET_UTIL_H
#define RIVULET_UTIL_H

#include "common.h"

#include <fstream>
#include <iostream>
#include <string>

using namespace std;

void memcpy_nts(void* dst, const void* src, size_t bytes);
void* am_memalign(size_t align, size_t size);
void am_free(void* ptr);

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

template <typename Pred>
void wait_for(Pred pred) {
  //volatile uint64_t duration = 1024;
  while(!pred()) {
    //for(uint64_t i=0; i<duration; i++);
    //duration *= 2;
  }
}

#endif