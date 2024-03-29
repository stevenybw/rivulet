#ifndef RIVULET_GARRAY_H
#define RIVULET_GARRAY_H

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>

#include "common.h"
#include "object.h"

using namespace std;

template <typename T>
struct GArray
{
  Object* _obj;
  T*      _data;
  size_t  _size;
  GArray(Object* obj) : _obj(obj) {
    _data = (T*) obj->local_data();
    assert(obj->local_capacity() % sizeof(T) == 0);
    _size = obj->local_capacity() / sizeof(T);
  }
  ~GArray() {
    delete _obj;
  }
  T& operator[](size_t idx) { return _data[idx]; }

  /*! \brief Get the communicator of this GArray
   */
  MPI_Comm communicator() {
    return _obj->communicator();
  }

  /*! \brief Get the partition id this process is in
   */
  int partition_id() {
    int rank;
    MPI_Comm_rank(_obj->communicator(), &rank);
    return rank;
  }

  /*! \brief Get the number of partitions this GArray has
   */
  int num_partitions() {
    int nprocs;
    MPI_Comm_size(_obj->communicator(), &nprocs);
    return nprocs;
  }

  template <typename AccT, typename AccOp, typename MergeOp>
  AccT accumulate(AccT init, int mpi_count, MPI_Datatype mpi_type, MPI_Op mpi_op, AccOp acc_op, MergeOp merge_op) {
    // get max node id for each partitions
    AccT acc_sum = init;
    uint64_t local_size = this->size();
    T* data = this->data();
    #pragma omp parallel
    {
      AccT local_acc_sum = init;
      #pragma omp for
      for (uint64_t i=0; i<local_size; i++) {
        local_acc_sum = acc_op(local_acc_sum, data[i]);
      }
      #pragma omp critical
      acc_sum = merge_op(acc_sum, local_acc_sum);
    }
    AccT global_acc_sum = init;
    MPI_Allreduce(&acc_sum, &global_acc_sum, mpi_count, mpi_type, mpi_op, _obj->communicator());
    return global_acc_sum;
  }
  
  size_t size() { return _size; }

  size_t global_size() {
    size_t global_size;
    MPI_Allreduce(&_size, &global_size, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, _obj->communicator());
    return global_size;
  }

  uint64_t global_checksum() {
    assert(sizeof(T) % sizeof(uint64_t) == 0);
    uint64_t local_checksum = 0;
    uint64_t* ptr = (uint64_t*) _data;
    uint64_t  num_elem = _size;
    for (size_t i=0; i<num_elem; i++) {
      local_checksum ^= ptr[i];
    }
    uint64_t global_checksum = 0;
    MPI_Allreduce(&local_checksum, &global_checksum, 1, MPI_UNSIGNED_LONG_LONG, MPI_BXOR, _obj->communicator());
    return global_checksum;
  }

  void resize(size_t size) {
    assert(size * sizeof(T) <= _obj->local_capacity());
    _size = size;
  }

  /*! \brief Append an element to the back of the GArray
   *
   */
  void push_back(const T& rhs) {
    assert(_size * sizeof(T) == _obj->local_capacity());
    _obj->resize((_size+1) * sizeof(T));
    _data = (T*) _obj->local_data();
    memcpy(&_data[_size], &rhs, sizeof(T));
    _size++;
  }

  void commit() { _obj->commit(); }

  T* data() { return _data; }
};

#endif
