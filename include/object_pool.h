#ifndef RIVULET_OBJECT_POOL_H
#define RIVULET_OBJECT_POOL_H

#include <cstring>
#include <random>
#include <string>

#include "common.h"
#include "file.h"
#include "object.h"

// enum { UNIFORMITY_UNIFORM_OBJECT=1, UNIFORMITY_SEPARATED_OBJECT=2 };
// enum { WRITABILITY_READ_ONLY=3, WRITABILITY_READ_WRITE=4 };

/*! \brief ObjectPool is responsible for creating & managing objects
 *
 *  ObjectPool keeps tracks on currently
 */
struct ObjectPool
{
  string dram_transient_prefix;
  string nvm_transient_prefix;

  random_device            _rand_dev;
  default_random_engine    _rand_gen;
  uniform_int_distribution<uint64_t> _rand_dist;

  ObjectPool(string dram_transient_prefix, string nvm_transient_prefix) : dram_transient_prefix(dram_transient_prefix), 
                                                                             nvm_transient_prefix(nvm_transient_prefix), 
                                                                             _rand_dev(),
                                                                             _rand_gen(_rand_dev()) {}

  string gen_anonymous_name() {
    char fname[256];
    snprintf(fname, 256, "%s_%016lx.bin", anonymous_prefix.c_str(), _rand_dist(_rand_gen));
    return string(fname);
  }

  // create a writable separate array with specified capacity
  Object* create_object(string obj_name, size_t capacity, ObjectMode mode) {
    assert(mode.is_separated());
    assert(mode.is_writable());
    bool is_anonymous;
    if (obj_name.size() == 0) {
      is_anonymous = true;
      obj_name = gen_anonymous_name();
    } else {
      is_anonymous = false;
    }
    MappedFile file;
    file.create(obj_name.c_str(), capacity);
    file.open(obj_name.c_str(), FILE_MODE_READ_WRITE, ACCESS_PATTERN_NORMAL);
    Object* obj = new Object([file]() mutable {file.close();});
    obj->is_anonymous = is_anonymous;
    obj->name = obj_name;
    obj->mode = mode;
    obj->data = file._addr;
    obj->capacity = capacity;

    return obj;
  }

  // load from a existing array
  Object* load_object(string obj_name, ObjectMode mode) {
    assert(obj_name.size() > 0);
    MappedFile file;
    if (mode.is_writable()) {
      file.open(obj_name.c_str(), FILE_MODE_READ_WRITE, ACCESS_PATTERN_NORMAL);
    } else {
      assert(mode.is_readonly());
      file.open(obj_name.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_NORMAL);
    }
    Object* obj = new Object([file]() mutable {file.close();});
    obj->is_anonymous = false;
    obj->name = obj_name;
    obj->mode = mode;
    obj->data = file._addr;
    obj->capacity = file._bytes;

    return obj;
  }
};


#endif