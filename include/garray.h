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
  void resize(size_t size) {
    assert(size * sizeof(T) <= _obj->local_capacity());
    _size = size;
  }
  T* data() { return _data; }
};

#endif
