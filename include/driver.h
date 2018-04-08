#ifndef RIVULET_GARRAY_DRIVER_H
#define RIVULET_GARRAY_DRIVER_H

#include "garray.h"
#include "graph.h"
#include "object_pool.h"

template <typename InputT, typename OutputT>
struct MapFn
{
  using InputType = InputT;
  using OutputType = OutputT;

  OutputT processElement(const InputT& in_element)=0;
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
  GArray<typename MapFnType::OutputType>* map(GArray<typename MapFnType::InputType>* input, const MapFnType& map_fn) {
    typename MapFnType::InputType* in_data_begin = input->local_begin();
    size_t num_local_elements = input->local_num_elements();    
    GArray<typename MapFnType::OutputType>* garr = create_array<typename MapFnType::OutputType>("", num_local_elements, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
    typename MapFnType::OutputType* out_data_begin = garr->local_begin();
    for(size_t i=0; i<num_local_elements; i++) {
      out_data_begin[i] = map_fn.processElement(in_data_begin[i]);
    }
    return garr;
  }

  template <typename T, typename PartitionFnType>
  GArray<T>* repartition(GArray<T>* input, const PartitionFnType& part_fn) {
    uint64_t duration;
    T* in_data_begin = input->local_begin();
    size_t num_local_elements = input->local_num_elements();
    REGION_BEGIN();
    sort(in_data_begin, in_data_begin+num_local_elements, [&part_fn](const T& lhs, const T& rhs) {
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
      MPI_AlltoallvL(input->local_begin(), scounts, sdispls, datatype, output->local_begin(), rcounts, rdispls, datatype, comm);
      REGION_END("Alltoallv for Output\n");
    }
    // post check
    {
      T* out_data_begin = output->local_begin();
      assert(output->local_num_elements() == recv_elements);
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
    GArray<pair<NodeT, NodeT>>* garr_tuples = create_array<pair<NodeT, NodeT>>(obj_name, graph.num_local_edges(), ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
    pair<NodeT, NodeT>* tuples = garr_tuples->local_begin();
    uint64_t curr_num_tuples = 0;
    for(uint64_t vid=graph.get_begin_vid(); vid<graph.get_end_vid(); vid++) {
      IndexT from_idx = graph.get_index_from_vid(vid);
      IndexT to_idx = graph.get_index_from_vid(vid+1);
      for(IndexT idx=from_idx; idx<to_idx; idx++) {
        uint64_t dst_vid = graph.get_edge_from_index[idx];
        tuples[curr_num_tuples].first = vid;
        tuples[curr_num_tuples].second = dst_vid;
        curr_num_tuples++;
      }
    }
    assert(curr_num_tuples == graph.num_local_edges());
    assert(garr_tuples->local_num_elements() == graph.num_local_edges());
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

//   template <typename NodeT, typename IndexT, typename PartFnType>
//   Graph<NodeT, IndexT>* make_graph_from_tuples(GArray<pair<NodeT, NodeT>>* garr_tuples, const PartFnType& part_fn, string name="") {
//     assert(garr_tuples->obj->mode.is_separated());
//     pair<NodeT, NodeT>* tuples = garr_tuples->local_begin();
//     uint64_t num_edges = garr_tuples->local_num_elements();
// 
//     string edges_name;
//     string index_name;
// 
//     if (name.size() == 0) {
//       edges_name = "";
//       index_name = "";
//     } else {
//       edges_name = name + ".edges";
//       index_name = name + ".index";
//     }
// 
//     GArray<NodeT>* garr_edges = create_array<NodeT>(edges_name, num_edges, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
// 
//     // verify the order of input tuples
//     for(uint64_t i=0; i<num_edges; i++) {
//       NodeT x = tuples[i].first;
//       NodeT y = tuples[i].second;
//       assert(part_fn(x) == obj_ctx.get_rank());
//       if (i != 0) {
//         if (tuples[i-1].first == tuples[i].first) {
//           assert(tuples[i-1].second <= tuples[i].second);
//         } else {
//           assert(tuples[i-1].first < tuples[i].first);
//         }
//       }
//       garr_edges[i] = y;
//     }
// 
//     uint64_t num_nodes = (tuples[num_edges-1].first) + 1;
//     GArray<IndexT>* garr_index = create_array<IndexT>(index_name, num_nodes+1, ObjectMode(UNIFORMITY_SEPARATED_OBJECT, WRITABILITY_READ_WRITE));
//     IndexT last_index = 0;
//     for (NodeT vid=0; vid<num_nodes; vid++) {
//       while(last_index<num_edges && tuples[last_index].first<vid) {
//         last_index++;
//       }
//       garr_index[vid] = last_index;
//     }
//     garr_index[num_nodes] = num_edges;
//     return new DistributedGraph<NodeT, IndexT>(std::move(edges), std::move(index));
//   }
};

#endif