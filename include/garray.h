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
