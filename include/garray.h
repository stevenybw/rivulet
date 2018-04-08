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
    _data = (T*) obj->data;
    assert(obj->capacity % sizeof(T) == 0);
    _size = obj->capacity / sizeof(T);
  }
  ~GArray() {
    delete _obj;
  }

  bool is_uniform() { return _obj->mode.is_uniform(); }
  bool is_separated() { return _obj->mode.is_separated(); }
  T& operator[](size_t idx) { return _data[idx]; }
  size_t size() { return _size; }
  T* data() { return _data; }
};

#endif
