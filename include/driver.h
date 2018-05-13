#ifndef RIVULET_GARRAY_DRIVER_H
#define RIVULET_GARRAY_DRIVER_H

#include <algorithm>
#include <stdexcept>
#include <tuple>
#include <parallel/algorithm>

#include <omp.h>

using namespace std;

#include "common.h"
#include "file.h"
#include "garray.h"
#include "graph.h"
// #include "object_pool.h"
#include "object.h"
#include "ympi.h"

/*! \brief Load configuration from environment variable
 */
struct Configuration {
  constexpr static char* NVM_OFF_CACHE_POOL_DIR = "NVM_OFF_CACHE_POOL_DIR";
  constexpr static char* NVM_OFF_CACHE_POOL_SIZE = "NVM_OFF_CACHE_POOL_SIZE";
  constexpr static char* NVM_ON_CACHE_POOL_DIR = "NVM_ON_CACHE_POOL_DIR";
  constexpr static char* NVM_ON_CACHE_POOL_SIZE = "NVM_ON_CACHE_POOL_SIZE";

  string nvm_off_cache_pool_dir;
  size_t nvm_off_cache_pool_size;
  string nvm_on_cahce_pool_dir;
  size_t nvm_on_cahce_pool_size;

  char* get_environment_var(const char* name) {
    char* result = getenv(name);
    if (result == NULL) {
      printf("[ERROR] Environment variable %s is required but does not set.\n", name);
    }
    return result;
  }

  Configuration() {
    nvm_off_cache_pool_dir = get_environment_var(NVM_OFF_CACHE_POOL_DIR);
    nvm_off_cache_pool_size = atoll(get_environment_var(NVM_OFF_CACHE_POOL_SIZE));
    nvm_on_cahce_pool_dir = get_environment_var(NVM_ON_CACHE_POOL_DIR);
    nvm_on_cahce_pool_size = atoll(get_environment_var(NVM_ON_CACHE_POOL_SIZE));
  }
};

/*! \brief Per-run context data
 */
struct ExecutionContext {
  // Path
  string dram_staging_path; // Not Used!
  string nvm_offcache_staging_path;
  string nvm_oncache_staging_path;

  // Random State
  random_device            _rand_dev;
  default_random_engine    _rand_gen;
  uniform_int_distribution<uint64_t> _rand_dist;

  // Communication
  MPI_Comm comm;
  int      rank;
  int      nprocs;

  ExecutionContext(string dram_staging_path, string nvm_offcache_staging_path, string nvm_oncache_staging_path, MPI_Comm comm) : 
              dram_staging_path(dram_staging_path), 
              nvm_offcache_staging_path(nvm_offcache_staging_path), 
              nvm_oncache_staging_path(nvm_oncache_staging_path),
              _rand_dev(),
              _rand_gen(_rand_dev()),
              comm(comm) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
  }

  /*! \brief Generate random number
   */
  uint64_t random_uint64() {
    return _rand_dist(_rand_gen);
  }

  string get_dram_prefix() { return dram_staging_path; }
  string get_nvm_offcache_prefix() { return nvm_offcache_staging_path; }
  string get_nvm_oncache_prefix() { return nvm_oncache_staging_path; }
  MPI_Comm get_comm() { return comm; }
  int get_rank() { return rank; }
  int get_nprocs() { return nprocs; }
};

/*! \brief <Not Used>Per-task request passed for each invocation
 *
 *  Per-task request includes inputs, outputs, and args
 *    inputs:  full object names for input objects to be loaded
 *    outputs: full object names for output objects to be stored
 *    args:    extra string arguments
 */
struct ExecutionRequest {
  int num_input;
  char* inputs[];
  int num_output;
  char* outputs[];
  int num_arg;
  char* args[];
};

/*! \brief ObjectRequirement is used to describe what kind of object is required to create
 *
 *  Object can be created as:
 *    1. Persist / Transient
 *    2. Memory / NVM
 *  Also, object can be loaded from persisted file from
 *    1. Memory / NVM
 *    2. ReadOnly / ReadWrite
 */
struct ObjectRequirement
{
  enum RequirementType
  {
    TYPE_CREATE = 0,
    TYPE_LOAD = 1
  };
  enum ObjectType
  {
    OBJECT_PERSIST = 0,
    OBJECT_TRANSIENT = 1
  };
  enum Capability
  {
    CAPABILITY_READONLY = 0,
    CAPABILITY_READWRITE = 1
  };
  RequirementType      _req_type;
  string               _fullname;
  ObjectType           _obj_type;
  Capability           _capability;
  Object::StorageLevel _storage_level;
  size_t               _init_capacity;

  RequirementType req_type() { return _req_type; }
  string fullname() { return _fullname; }
  ObjectType obj_type() { return _obj_type; }
  Object::StorageLevel storage_level() { return _storage_level; }

  bool is_type_create() { return _req_type == TYPE_CREATE; }
  bool is_transient() { return _obj_type == OBJECT_TRANSIENT; } /**< Whether the required object is transient */
  bool is_in_memory() { return _storage_level == Object::MEMORY; } /**< Whether the required object is in DRAM */

  size_t local_capacity() { return _init_capacity; }
  void set_local_capacity(size_t capacity) { _init_capacity = capacity; }

  /*! \brief Load the object from given fullname
   *
   *  For now, we set the constraint that loaded object must be read only for crash consistency.
   */
  static ObjectRequirement load_from(string fullname) {
    ObjectRequirement obj_req;
    obj_req._req_type  = TYPE_LOAD;
    obj_req._fullname = fullname;
    obj_req._obj_type  = OBJECT_TRANSIENT;
    obj_req._capability= CAPABILITY_READONLY;
    return obj_req;
  }

  static ObjectRequirement create_transient(Object::StorageLevel storage_level, size_t init_capacity = 0) {
    ObjectRequirement obj_req;
    obj_req._req_type   = TYPE_CREATE;
    obj_req._fullname  = "";
    obj_req._obj_type   = OBJECT_TRANSIENT;
    obj_req._capability = CAPABILITY_READWRITE;
    obj_req._storage_level = storage_level;
    obj_req._init_capacity = init_capacity;
    return obj_req;
  }

  static ObjectRequirement create_persist(string fullname, size_t init_capacity = 0) {
    ObjectRequirement obj_req;
    obj_req._req_type   = TYPE_CREATE;
    obj_req._fullname  = fullname;
    obj_req._obj_type   = OBJECT_PERSIST;
    obj_req._capability = CAPABILITY_READWRITE;
    obj_req._init_capacity = init_capacity;
    return obj_req;
  }
};

template< class RandomIt, class Compare >
void my_sort( RandomIt first, RandomIt last, Compare comp ) {
  uint64_t duration = -currentTimeUs();
  /*
  #pragma omp parallel
  {
    int thread_id   = omp_get_thread_num();
    int num_threads = omp_get_num_threads();
    size_t nel = last - first;
    size_t nel_chunk = nel / num_threads;
    RandomIt begin_it = first + (thread_id * nel_chunk);
    RandomIt end_it = first + ((thread_id == num_threads-1)?nel:(thread_id+1)*nel_chunk);
    sort(begin_it, end_it, comp);
  }
  duration += currentTimeUs();
  printf("  parallel pre-sort time = %lf sec\n", 1e-6 * duration);
  */
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  duration = -currentTimeUs();
  // TODO __gnu_parallel::sort leads to segfault in MPI_Finalize()
  //__gnu_parallel::sort(first, last, comp);
  sort(first, last, comp);
  duration += currentTimeUs();
  printf("  %d> serial global sort time = %lf sec\n", rank, 1e-6 * duration);
}


template <typename InputT, typename OutputT>
struct MapFn
{
  using InputType = InputT;
  using OutputType = OutputT;

  virtual OutputT processElement(const InputT& in_element)=0;
};

/*! \brief Front end driver that interacts with users
 *
 *  Driver is responsible for loading, creating and manipulating GArray.
 */
struct Driver
{
  ExecutionContext& ctx;
  MPI_Comm comm;
  int rank;
  int nprocs;
  Driver(ExecutionContext& ctx) : ctx(ctx), comm(ctx.get_comm()), rank(ctx.get_rank()), nprocs(ctx.get_nprocs()) {
  }

  string storage_level_to_prefix(Object::StorageLevel storage_level)
  {
    if (storage_level == Object::NVM_OFF_CACHE) {
      return ctx.get_nvm_offcache_prefix();
    } else if (storage_level == Object::NVM_ON_CACHE) {
      return ctx.get_nvm_oncache_prefix();
    } else {
      throw invalid_argument("Unknown storage level");
    }
  }

  /*! \brief Create a new array
   *
   *  Requirement type can be: TYPE_CREATE or TYPE_LOAD
   *  TYPE_CREATE
   *    PERSIST: To create a persist object, a path must be specified and it will be overwritten afterwards
   *    TRANSIENT: Allocated via malloc if MEMORY, otherwise create temporary file at corresponding directory
   *  TYPE_LOAD
   *    PERSIST: A path must be specified
   */
  template <typename T>
  GArray<T>* create_array(ObjectRequirement obj_req, size_t new_size = 0) {
    if (obj_req.req_type() == ObjectRequirement::TYPE_CREATE) {
      if (new_size > 0) {
        obj_req.set_local_capacity(new_size * sizeof(T));
      }
      size_t local_capacity = obj_req.local_capacity();
      if (obj_req.is_transient() && obj_req.is_in_memory()) {
        printf("%d> Create transient DRAM array (bytes = %zu)\n", rank, local_capacity);
        void* local_data = malloc(local_capacity);
        Object* obj = new Object([](){;}, [local_data](){free(local_data);});
        obj->_comm = ctx.get_comm();
        obj->_is_persist = false;
        obj->_fullname = "<memory>";
        obj->_local_data = malloc(local_capacity);
        obj->_local_capacity = local_capacity;
        GArray<T>* garr = new GArray<T>(obj);
        garr->resize(new_size);
        return garr;
      }
      bool is_persist;
      string fullname;
      MappedFile file;
      if (obj_req.obj_type() == ObjectRequirement::OBJECT_TRANSIENT) {
        is_persist = false;
        string prefix = storage_level_to_prefix(obj_req.storage_level());
        char buf[256];
        snprintf(buf, 256, "%s/temp_%016lx.bin", prefix.c_str(), ctx.random_uint64());
        fullname = buf;
        file.create(fullname.c_str(), local_capacity);
        file.open(fullname.c_str(), FILE_MODE_READ_WRITE, ACCESS_PATTERN_NORMAL);
        file.unlink();
      } else if (obj_req.obj_type() == ObjectRequirement::OBJECT_PERSIST) {
        is_persist = true;
        fullname = obj_req.fullname();
        file.create(fullname.c_str(), local_capacity);
        file.open(fullname.c_str(), FILE_MODE_READ_WRITE, ACCESS_PATTERN_NORMAL);
      } else {
        throw invalid_argument("Unknown object type");
      }
      Object* obj;
      if (is_persist) {
        printf("%d> Create persist array (path = %s  bytes = %zu)\n", rank, file._path.c_str(), local_capacity);
        obj = new Object([file]() mutable {file.msync();}, [file]() mutable {file.close();});
      } else {
        printf("%d> Create transient file (path = %s  bytes = %zu)\n", rank, file._path.c_str(), local_capacity);
        // for transient object, destruction implies an unlink
        obj = new Object([file]() mutable {file.msync();}, [file]() mutable {file.close();});
      }
      obj->_comm = ctx.get_comm(); 
      obj->_is_persist = is_persist;
      obj->_fullname = fullname;
      obj->_local_data = file._addr;
      obj->_local_capacity = local_capacity;
      GArray<T>* garr = new GArray<T>(obj);
      garr->resize(new_size); 
      return garr;
    } else if (obj_req.req_type() == ObjectRequirement::TYPE_LOAD) {
      string fullname = obj_req.fullname();
      fullname = obj_req.fullname(); 
      MappedFile file;
      file.open(fullname.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_NORMAL);
      Object* obj = new Object([file]() mutable {file.msync();}, [file]() mutable {file.close();});
      obj->_comm  = ctx.get_comm();
      obj->_is_persist = true;
      obj->_fullname   = fullname;
      obj->_local_data = file.get_addr();
      obj->_local_capacity = file.get_bytes(); 
      assert(file.get_bytes() % sizeof(T) == 0);
      GArray<T>* garr = new GArray<T>(obj);
      garr->resize(file.get_bytes() / sizeof(T));
      return garr;
    } else {
      assert(false);
    }
  }

  /*! \brief Read from binary records
   *
   *  The file path must be in NFS
   */
  template <typename T>
  GArray<T>* readFromBinaryRecords(string path, size_t record_size = sizeof(T)) {
    int rank = ctx.get_rank();
    int nprocs = ctx.get_nprocs();
    MappedFile file;
    file.open(path.c_str(), FILE_MODE_READ_ONLY, ACCESS_PATTERN_NORMAL);
    assert(file.get_bytes() % record_size == 0);
    size_t num_elements = file.get_bytes() / record_size;
    size_t num_elements_chunk = num_elements / nprocs;
    size_t from_idx = rank * num_elements_chunk;
    size_t to_idx = (rank == (nprocs-1))?num_elements:(rank+1)*num_elements_chunk;
    T* data = (T*) file.get_addr();
    Object* obj = new Object([](){}, [file]() mutable {file.close();});
    obj->_comm = ctx.get_comm();
    obj->_is_persist = true;
    obj->_fullname   = "_";
    obj->_local_data = &data[from_idx*(record_size/sizeof(T))];
    obj->_local_capacity = (to_idx - from_idx) * record_size;
    GArray<T>* garr = new GArray<T>(obj);
    return garr;
  }

  /*! \brief Map operation
   *
   *  input: GArray<T>
   *  map_fn: MapFnType, must inherit MapFn 
   *  obj_req: Requirement for the object
   */
  template <typename MapFnType>
  GArray<typename MapFnType::OutputType>* map(GArray<typename MapFnType::InputType>* input, MapFnType& map_fn, ObjectRequirement obj_req) {
    assert(obj_req.is_type_create());
    typename MapFnType::InputType* in_data_begin = input->data(); 
    size_t num_local_elements = input->size(); 
    GArray<typename MapFnType::OutputType>* garr = create_array<typename MapFnType::OutputType>(obj_req, num_local_elements);
    typename MapFnType::OutputType* out_data_begin = garr->data();
    #pragma omp parallel for
    for(size_t i=0; i<num_local_elements; i++) {
      out_data_begin[i] = map_fn.processElement(in_data_begin[i]);
    }
    return garr;
  }

  /*! \brief Repartition operation
   *
   *  input: GArray<T>
   *  part_fn: PartitionFnType, must be a function from T to int
   *  obj_req: Requirement for the object
   */
  template <typename T, typename PartitionFnType>
  GArray<T>* repartition(GArray<T>* input, const PartitionFnType& part_fn, ObjectRequirement obj_req) {
    assert(obj_req.is_type_create());
    uint64_t duration;
    T* in_data_begin = input->data();
    size_t num_local_elements = input->size();
    GArray<T>* med = create_array<T>(ObjectRequirement::create_transient(Object::MEMORY), num_local_elements);
    T* med_data_begin = med->data();
    LOG_BEGIN();
    LOG_INFO("Copy the array to be sorted");
    memcpy(med_data_begin, in_data_begin, num_local_elements * sizeof(T));

    LOG_INFO("Qsort for Ordering Tuples by Rank");
    my_sort(med_data_begin, med_data_begin+num_local_elements, [&part_fn](const T& lhs, const T& rhs) {
      int lhs_part = part_fn(lhs);
      int rhs_part = part_fn(rhs);
      return lhs_part < rhs_part;
    });
    
    // check for order
    for(size_t i=0; i<num_local_elements-1; i++) {
      assert(part_fn(med_data_begin[i]) < nprocs);
      assert(part_fn(med_data_begin[i+1]) < nprocs);
      assert(part_fn(med_data_begin[i]) <= part_fn(med_data_begin[i+1]));
    }
    size_t sdispls[nprocs];
    size_t scounts[nprocs];
    size_t rcounts[nprocs];
    size_t rdispls[nprocs];
    size_t recv_elements = 0;
    GArray<T>* output = NULL;
    {
      size_t i=0;
      for (int curr_rank=0; curr_rank<nprocs; curr_rank++) {
        while (i < num_local_elements && part_fn(med_data_begin[i]) < curr_rank) {
          i++;
        }
        sdispls[curr_rank] = i;
      }
      for (int curr_rank=0; curr_rank<nprocs-1; curr_rank++) {
        scounts[curr_rank] = sdispls[curr_rank+1] - sdispls[curr_rank];
      }
      scounts[nprocs-1] = num_local_elements - sdispls[nprocs-1];
      MPI_Alltoall(scounts, 1, MPI_UNSIGNED_LONG_LONG, rcounts, 1, MPI_UNSIGNED_LONG_LONG, comm);
      rdispls[0] = 0;
      for (int curr_rank=0; curr_rank<nprocs-1; curr_rank++) {
        rdispls[curr_rank+1] = rdispls[curr_rank] + rcounts[curr_rank];
      }
      recv_elements = rdispls[nprocs-1] + rcounts[nprocs-1];
      output = create_array<T>(obj_req, recv_elements);
    }
    T* out_data_begin = output->data();
    MPI_Datatype datatype;
    MPI_Type_contiguous(sizeof(T), MPI_CHAR, &datatype);
    MPI_Type_commit(&datatype);
    LOG_INFO("Alltoallv for Output\n");
    YMPI_AlltoallvL(med_data_begin, scounts, sdispls, datatype, out_data_begin, rcounts, rdispls, datatype, comm);
    // post check
    {
      assert(output->size() == recv_elements);
      size_t total_num_input_element;
      size_t total_num_output_element;
      MPI_Allreduce(&num_local_elements, &total_num_input_element, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
      MPI_Allreduce(&recv_elements, &total_num_output_element, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, comm);
      assert(total_num_input_element == total_num_output_element);
      for(size_t i=0; i<recv_elements; i++) {
        assert(part_fn(out_data_begin[i]) == rank);
      }
    }
    delete med;
    LOG_INFO("Repartition Done");
    return output;
  }

  // template <typename NodeT, typename IndexT>
  // GArray<pair<NodeT, NodeT>>* to_tuples(SharedGraph<NodeT, IndexT>& graph, string obj_name = "") {
  //   uint64_t local_num_nodes = graph.local_num_nodes();
  //   uint64_t local_num_edges = graph.local_num_edges();
  //   uint64_t begin_vid = graph.get_begin_vid();
  //   uint64_t end_vid = graph.get_end_vid();
  //   printf("  %d> local_num_nodes = %lu\n", rank, local_num_nodes);
  //   printf("  %d> local_num_edges = %lu\n", rank, local_num_edges);
  //   printf("  %d> begin_vid = %lu\n", rank, begin_vid);
  //   printf("  %d> end_vid = %lu\n", rank, end_vid);
  //   GArray<pair<NodeT, NodeT>>* garr_tuples = create_array<pair<NodeT, NodeT>>(obj_name, graph.local_num_edges(), ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
  //   pair<NodeT, NodeT>* tuples = garr_tuples->data();
  //   uint64_t partition_begin_index = graph.get_index_from_vid(graph.get_begin_vid());
  //   // uint64_t curr_num_tuples = 0;
  //   #pragma omp parallel for
  //   for(uint64_t vid=graph.get_begin_vid(); vid<graph.get_end_vid(); vid++) {
  //     IndexT from_idx = graph.get_index_from_vid(vid);
  //     IndexT to_idx = graph.get_index_from_vid(vid+1);
  //     for(IndexT idx=from_idx; idx<to_idx; idx++) {
  //       uint64_t dst_vid = graph.get_edge_from_index(idx);
  //       uint64_t local_idx = idx - partition_begin_index;
  //       assert(local_idx >= 0);
  //       assert(local_idx < garr_tuples->size());
  //       tuples[local_idx].first = vid;
  //       tuples[local_idx].second = dst_vid;
  //       // curr_num_tuples++;
  //     }
  //   }
  //   // assert(curr_num_tuples == graph.local_num_edges());
  //   assert(garr_tuples->size() == graph.local_num_edges());
  //   return garr_tuples;
  // }

  // Construct the graph given edges and index
  // template <typename NodeT, typename IndexT>
  // Graph<NodeT, IndexT>* make_graph_from_csr(GArray<NodeT>* edges, GArray<IndexT>* index) {
  //   assert(edges->obj->mode == index->obj->mode);
  //   bool is_uniform = edges->is_uniform();
  //   if (is_uniform) {
  //     return new SharedGraph<NodeT, IndexT>(edges, index);
  //   } else {
  //     return new DistributedGraph<NodeT, IndexT>(edges, index);
  //   }
  // }

  template <typename NodeT, typename IndexT, typename PartFnType, typename OffsetFnType>
  tuple<GArray<NodeT>*, GArray<IndexT>*> make_csr_from_tuples(GArray<pair<NodeT, NodeT>>* garr_tuples, const PartFnType& part_fn, const OffsetFnType& offset_fn, ObjectRequirement edges_obj_req, ObjectRequirement index_obj_req) {
    assert(edges_obj_req.is_type_create());
    assert(index_obj_req.is_type_create());
    pair<NodeT, NodeT>* tuples = garr_tuples->data();
    uint64_t num_edges = garr_tuples->size();
    
    GArray<NodeT>* garr_edges = create_array<NodeT>(edges_obj_req, num_edges);
    NodeT* edges = garr_edges->data();
    // verify the order of input tuples
    #pragma omp parallel for
    for(uint64_t i=0; i<num_edges; i++) {
      NodeT x = tuples[i].first;
      NodeT y = tuples[i].second;
      assert(part_fn(x) == rank);
      if (i != 0) {
        if (tuples[i-1].first == tuples[i].first) {
          assert(tuples[i-1].second <= tuples[i].second);
        } else {
          assert(tuples[i-1].first < tuples[i].first);
        }
      }
      edges[i] = y;
    }
    // largest node id in this partition
    uint64_t largest_offset = std::max(offset_fn(tuples[num_edges-1].first), offset_fn(tuples[num_edges-1].second));
    uint64_t num_nodes = largest_offset + 1;
    GArray<IndexT>* garr_index = create_array<IndexT>(index_obj_req, num_nodes+1);
    IndexT* index = garr_index->data();
    IndexT last_index = 0;
    for (NodeT vid=0; vid<num_nodes; vid++) {
      while(last_index<num_edges && offset_fn(tuples[last_index].first)<vid) {
        last_index++;
      }
      index[vid] = last_index;
    }
    index[num_nodes] = num_edges;
    //if (last_index != num_edges) {
    //  printf("[WARNING] last_index = %lu, num_edges = %lu\n", last_index, num_edges);
    //}
    return make_tuple(garr_edges, garr_index);
  }
};

#endif
