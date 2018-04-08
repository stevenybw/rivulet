#ifndef RIVULET_OBJECT_H
#define RIVULET_OBJECT_H

#include <functional>
#include <string>

using namespace std;

enum { UNIFORMITY_UNIFORM_OBJECT=1, UNIFORMITY_SEPARATED_OBJECT=2 };
enum { WRITABILITY_READ_ONLY=3, WRITABILITY_READ_WRITE=4 };

struct ObjectMode
{
  int uniformity;
  int writability;
  ObjectMode() : uniformity(-1), writability(-1) {}
  ObjectMode(int uniformity, int writability) : uniformity(uniformity), writability(writability) {}

  bool is_uniform() { return uniformity == UNIFORMITY_UNIFORM_OBJECT; }
  bool is_separated() { return uniformity == UNIFORMITY_SEPARATED_OBJECT; }
  bool is_writable() { return writability == WRITABILITY_READ_WRITE; }
  bool is_readonly() { return writability == WRITABILITY_READ_ONLY; }
};

struct Object
{
  bool is_anonymous;
  string name;
  ObjectMode mode;
  void* data;
  size_t capacity;
  function<void()> on_delete;

  Object(const function<void()>& on_delete) : on_delete(on_delete) {}

  ~Object() {
    on_delete();
  }
};

#endif