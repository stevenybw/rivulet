#ifndef RIVULET_GARRAY_DRIVER_H
#define RIVULET_GARRAY_DRIVER_H

#include <tuple>
#include <parallel/algorithm>

#include <omp.h>

using namespace std;

#include "garray.h"
#include "graph.h"
#include "object_pool.h"
#include "ympi.h"

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
  __gnu_parallel::sort(first, last, comp);
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

struct Driver
{
  int rank;
  int nprocs;
  MPI_Comm    comm;
  ObjectPool* obj_pool;

  int get_rank(); // TODO
  int get_nprocs(); // TODO
  MPI_Comm get_comm(); // TODO

  Driver(MPI_Comm comm, ObjectPool* obj_pool) : comm(comm), obj_pool(obj_pool) {
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nprocs);
  }

  template <typename T>
  GArray<T>* create_array(string obj_name, size_t num_elements, ObjectMode mode) {
    Object* obj = obj_pool->create_object(obj_name, num_elements * sizeof(T), mode);
    GArray<T>* garr = new GArray<T>(obj);
    return garr;
  }

  template <typename T>
  GArray<T>* load_array(string obj_name, ObjectMode mode) {
    Object* obj = obj_pool->load_object(obj_name, mode);
    assert(obj->capacity % sizeof(T) == 0);
    GArray<T>* garr = new GArray<T>(obj);
    return garr;
  }

  template <typename MapFnType>
  GArray<typename MapFnType::OutputType>* map(GArray<typename MapFnType::InputType>* input, MapFnType& map_fn) {
    assert(input->is_separated());
    typename MapFnType::InputType* in_data_begin = input->data();
    size_t num_local_elements = input->size();
    GArray<typename MapFnType::OutputType>* garr = create_array<typename MapFnType::OutputType>("", num_local_elements, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
    typename MapFnType::OutputType* out_data_begin = garr->data();
    #pragma omp parallel for
    for(size_t i=0; i<num_local_elements; i++) {
      out_data_begin[i] = map_fn.processElement(in_data_begin[i]);
    }
    return garr;
  }

  template <typename T, typename PartitionFnType>
  GArray<T>* repartition(GArray<T>* input, const PartitionFnType& part_fn) {
    assert(input->is_separated());
    uint64_t duration;
    T* in_data_begin = input->data();
    size_t num_local_elements = input->size();
    REGION_BEGIN();
    my_sort(in_data_begin, in_data_begin+num_local_elements, [&part_fn](const T& lhs, const T& rhs) {
      int lhs_part = part_fn(lhs);
      int rhs_part = part_fn(rhs);
      return lhs_part < rhs_part;
    });
    REGION_END("Qsort for Ordering Tuples by Rank");
    // check for order
    for(size_t i=0; i<num_local_elements-1; i++) {
      assert(part_fn(in_data_begin[i]) < nprocs);
      assert(part_fn(in_data_begin[i+1]) < nprocs);
      assert(part_fn(in_data_begin[i]) <= part_fn(in_data_begin[i+1]));
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
        while (i < num_local_elements && part_fn(in_data_begin[i]) < curr_rank) {
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
      output = create_array<T>("", recv_elements, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
      MPI_Datatype datatype;
      MPI_Type_contiguous(sizeof(T), MPI_CHAR, &datatype);
      MPI_Type_commit(&datatype);
      REGION_BEGIN();
      YMPI_AlltoallvL(input->data(), scounts, sdispls, datatype, output->data(), rcounts, rdispls, datatype, comm);
      REGION_END("Alltoallv for Output\n");
    }
    // post check
    {
      T* out_data_begin = output->data();
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
    return output;
  }

  template <typename NodeT, typename IndexT>
  GArray<pair<NodeT, NodeT>>* to_tuples(SharedGraph<NodeT, IndexT>& graph, string obj_name = "") {
    uint64_t local_num_nodes = graph.local_num_nodes();
    uint64_t local_num_edges = graph.local_num_edges();
    uint64_t begin_vid = graph.get_begin_vid();
    uint64_t end_vid = graph.get_end_vid();
    printf("  %d> local_num_nodes = %lu\n", rank, local_num_nodes);
    printf("  %d> local_num_edges = %lu\n", rank, local_num_edges);
    printf("  %d> begin_vid = %lu\n", rank, begin_vid);
    printf("  %d> end_vid = %lu\n", rank, end_vid);
    GArray<pair<NodeT, NodeT>>* garr_tuples = create_array<pair<NodeT, NodeT>>(obj_name, graph.local_num_edges(), ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
    pair<NodeT, NodeT>* tuples = garr_tuples->data();
    uint64_t partition_begin_index = graph.get_index_from_vid(graph.get_begin_vid());
    // uint64_t curr_num_tuples = 0;
    #pragma omp parallel for
    for(uint64_t vid=graph.get_begin_vid(); vid<graph.get_end_vid(); vid++) {
      IndexT from_idx = graph.get_index_from_vid(vid);
      IndexT to_idx = graph.get_index_from_vid(vid+1);
      for(IndexT idx=from_idx; idx<to_idx; idx++) {
        uint64_t dst_vid = graph.get_edge_from_index(idx);
        uint64_t local_idx = idx - partition_begin_index;
        assert(local_idx >= 0);
        assert(local_idx < garr_tuples->size());
        tuples[local_idx].first = vid;
        tuples[local_idx].second = dst_vid;
        // curr_num_tuples++;
      }
    }
    // assert(curr_num_tuples == graph.local_num_edges());
    assert(garr_tuples->size() == graph.local_num_edges());
    return garr_tuples;
  }

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
  tuple<GArray<NodeT>*, GArray<IndexT>*> make_csr_from_tuples(GArray<pair<NodeT, NodeT>>* garr_tuples, const PartFnType& part_fn, const OffsetFnType& offset_fn, string name="") {
    assert(garr_tuples->is_separated());
    pair<NodeT, NodeT>* tuples = garr_tuples->data();
    uint64_t num_edges = garr_tuples->size();
    
    string edges_name;
    string index_name;
    
    if (name.size() == 0) {
      edges_name = "";
      index_name = "";
    } else {
      edges_name = name + ".edges";
      index_name = name + ".index";
    }
    
    GArray<NodeT>* garr_edges = create_array<NodeT>(edges_name, num_edges, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
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
    uint64_t largest_offset = offset_fn(tuples[num_edges-1].first);
    uint64_t num_nodes = largest_offset + 1;
    GArray<IndexT>* garr_index = create_array<IndexT>(index_name, num_nodes+1, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
    IndexT* index = garr_index->data();
    IndexT last_index = 0;
    for (NodeT vid=0; vid<num_nodes; vid++) {
      while(last_index<num_edges && offset_fn(tuples[last_index].first)<vid) {
        last_index++;
      }
      index[vid] = last_index;
    }
    index[num_nodes] = num_edges;
    if (last_index != num_edges) {
      printf("[WARNING] last_index = %lu, num_edges = %lu\n", last_index, num_edges);
    }
    return make_tuple(garr_edges, garr_index);
  }
};

#endif