#ifndef RIVULET_OBJECT_H
#define RIVULET_OBJECT_H

#include <functional>
#include <string>

using namespace std;

/*! \brief Object is a distributed-stored array
 *
 *  An object is a named and mmapped region that stores a contiguous subset of data (called **local data**) of the global view.
 */
class Object
{
  friend class Driver;

  MPI_Comm _comm;
  bool     _is_persist;
  string   _fullname;
  void*    _local_data;
  size_t   _local_capacity;
  function<void()> _on_commit;
  function<void()> _on_delete;

public:
  /*! \brief Object can be in different storage level that indicates its storage position
   *
   *  MEMORY: This object is in DRAM
   *  NVM_OFF_CACHE: This object is in NVM, and page cache is off (for example, DAX)
   *  NVM_ON_CACHE: This object is in NVM, and page cache is on (for example, conventional block device)
   */
  enum StorageLevel {
    MEMORY=0,
    NVM_OFF_CACHE=1,
    NVM_ON_CACHE=2
  };

  /*! \brief ObjectState indicates the persist state of this object
   *
   *  An object can either in CLEAN state or DIRTY state. CLEAN object is guarantee to be
   *  consistent upon crash but read-only. DIRTY object is corrupted after crash.
   */
  enum ObjectState {
    CLEAN=0,
    DIRTY=1
  };

  Object(const function<void()>& on_commit, const function<void()>& on_delete) : _on_commit(on_commit), _on_delete(on_delete) {}

  ~Object() {
    _on_delete();
  }

  /*! \brief Return the communicator this object is in
   */
  MPI_Comm communicator() { return _comm; };

  /*! \brief Indicates if the object is persist (return true) or transient (return false)
   *
   *  A transient object will be removed from storage immediately upon destruction
   *  A persist object will be flushed into devices upon commit
   */
  bool is_persist() { return _is_persist; }

  /*! \brief [GroupOp] Commit the change to underlying storage
   *
   *  Apply the changes into underlying storage. This will switch the object from DIRTY state
   *  to CLEAN state. It will block until all the processes in the communicator have committed.
   */
  void commit() {
    _on_commit();
    MPI_Barrier(_comm);
  }

  /*! \brief Get the local data buffer of this object
   */
  void* local_data() { return _local_data; }

  /*! \brief Get the capacity in bytes of local data buffer
   */
  size_t local_capacity() { return _local_capacity; }
};

#endif