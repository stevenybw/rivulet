#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <string>
#include <thread>

using Thread = std::thread;

// TODO: Channel mechanis,
using Channel = ...;
using InputChannel = ...;
using OutputChannel = ...;
using string = std::string;

using InputChannelList = std::vector<InputChannel*>;
using OutputChannelList = std::vector<OutputChannel*>;

// using ChannelList = std::vector<Channel*>;

template <typename T>
struct TS {
  uint64_t ts;
  T e;
};

template <typename T>
struct WIN {
  WIN(T&& e, uint64_t id) : e(std::move(e)), id(id) {}
  uint64_t id;
  T e;
};

struct WorkerDescriptor {
  int worker_rank;
  int worker_lid;
  int worker_socket_id;
  int rank() { return worker_rank; }
  int lid() { return worker_lid; }
  int socket_id() { return worker_socket_id; }
};

struct Device {
  CPUDevice* CPU() { return new CPUDevice; }
};

struct CPUDevice : Device {
  std::set<int> node_set; // ranks to distributed to
  std::set<int> socket_set; // socket to distributed
  int num_proc_per_socket;

  CPUDevice* all_nodes() {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    for(int i=0; i<nprocs; i++) {
      node_set.push_back(i);
    }
    return this;
  }

  CPUDevice* all_sockets() {
    int num_sockets = numa_num_configured_nodes();
    for(int i=0; i<num_sockets; i++) {
      socket_set.push_back(i);
    }
    return this;
  }

  CPUDevice* cpu_per_socket(int num_cpus) {
    num_proc_per_socket = num_cpus;
    return this;
  }
};

CPUDevice* g_curr_device;

void WithDevice(CPUDevice* device) {
  g_curr_device = device;
}

// A PTransform instance bound with a unique id and optionally a state
struct PTInstance {
  int instance_id;
  PTransform* ptransform;
  void* state;
};

struct PTransform {
  string     name;
  CPUDevice* device;
  PTransform() : name(""), device(g_curr_device) {}
  PTransform* set_name(const string& name) { this->name = name; return this; }
  string display_name() { return name + "@" + type_name(); }

  virtual string type_name()=0;
  virtual bool is_elementwise()=0; // hint for scheduler
  virtual bool is_shuffle()=0; // hint for shceduler
  virtual PTInstance* create_instance(int instance_id)=0;
  virtual void execute(const InputChannelList& in, const OutputChannelList& out, void* state)=0; // execute this instance until it has finished
  // virtual bool progress(const InputChannelList& in, const OutputChannelList& out, void* state)=0; // return false to report completion
};

template <typename InputT, typename OutputT>
struct TaggedPTransform : public PTransform {};

struct PCollectionInput {};  // placeholder type for input of source
struct PCollectionOutput {}; // placeholder type for output of sink

struct CollectionOutputEntry 
{
  PCollection* target_collection;
  PTransform*  via_transform;

  CollectionOutputEntry() : target_collection(NULL), via_transform(NULL) {}
};

using CollectionOutputEntryList = std::vector<CollectionOutputEntry>;

template <typename T>
struct PCollection 
{
  PCollection() {}

  CollectionOutputEntryList outputs;

  template <typename InputT, typename OutputT>
  PCollection<OutputT>* apply(TaggedPTransform<InputT, OutputT>* pt) {
    PCollection<OutputT>* result = new PCollection<OutputT>;
    CollectionOutputEntry entry;
    entry.target_collection = result;
    entry.via_transform = pt;
    outputs.push_back(entry);
    return result;
  }
};

// Scheduler fo a pipeline (assume a strait pipeline, no diverge), and support
// distributed environment.
struct DistributedPipelineScheduler {
  // Stage: CollectionOutputEntries's via_transform
  struct Stage 
  {
    Stage() : device(NULL) {}
    // @TODO add move constructor
    std::vector<CollectionOutputEntry> pt_list;
    std::vector<WorkerDescriptor> workers;
    std::vector<WorkerDescriptor> local_workers;
    CPUDevice* device;
    void set_device(CPUDevice* device) {
      assert(device == NULL);
      this->device = device;
      int rank, nprocs;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

      const std::set<int>& node_set = device->node_set;
      const std::set<int>& socket_set = device->socket_set;
      int num_proc_per_socket = device->num_proc_per_socket;
      for (int node_id : node_set) {
        int lid = 0;
        for (int socket_id : socket_set) {
          for (int i=0; i<num_proc_per_socket; i++) {
            WorkerDescriptor wd;
            wd.worker_rank = node_id;
            wd.worker_lid  = lid++;
            wd.worker_socket_id = socket_id;
            workers.push_back(wd);
            if (node_id == rank) {
              local_workers.push_back(wd);
            }
          }
        }
      }
    }
    void add_transform(CollectionOutputEntry e)    { pt_list.push_back(e); }
    int num_local_workers()    { return local_workers.size(); }
    vector<WorkerDescriptor>& get_local_workers()    { return local_workers; }
    vector<WorkerDescriptor>& get_workers()    { return workers; }
  };

  struct Task 
  {
    int id; // unique id in a stage for this task
    int bind_socket_id; // socket id to bound to
    bool launched;
    std::vector<Thread> worker_thread_list; // one thread for each task transform
    Task(int task_id) : id(task_id), bind_socket_id(-1), launched(false) {}
    int bind_socket(int bind_socket_id) {
      this->bind_socket_id = bind_socket_id;
    }
    InputChannelList stage_input_channels; // from last stage
    std::vector<LocalOutputChannel*> intermediate_output_channels; // intra-stage
    std::vector<LocalInputChannel*>  intermediate_input_channels; // intra-stage
    OutputChannelList stage_output_channels; // to next stage
    std::vector<PTInstance*> transform_instance_list;

    int num_transforms() { return transform_instance_list.size(); }

    void launch() {
      assert(!launched);
      launched = true;
      assert(transform_instance_list.size() > 0);

      if (transform_instance_list.size() == 1) {
        worker_thread_list.push_back(std::move(Thread([this](){
          if (bind_socket_id > 0) {
            numa_run_on_node(bind_socket_id);
          }
          PTInstance* instance = transform_instance_list[0];
          PTransform* ptransform = instance->ptransform;
          void* state = instance->state;
          ptransform->execute(stage_input_channels, stage_output_channels, state);
        })));
      } else {
        for(int i=0; i<transform_instance_list.size(); i++) {
          worker_thread_list.push_back(std::move(Thread([this, i](){
            if (bind_socket_id > 0) {
              numa_run_on_node(bind_socket_id);
            }
            PTInstance* pti = transform_instance_list[i];
            PTransform* pt  = pti->ptransform;
            void* state     = pti->state;
            InputChannelList  tmp_icl;
            OutputChannelList tmp_ocl;
            tmp_icl.push_back(NULL);
            tmp_ocl.push_back(NULL);
            if (i == 0) {
              tmp_ocl[0] = intermediate_output_channels[i];
              pt->execute(stage_input_channels, tmp_ocl, state);
            } else if (i == (transform_instance_list.size() - 1)) {
              tmp_icl[0] = intermediate_input_channels[i-1];
              pt->execute(tmp_icl, stage_output_channels, state);
            } else {
              tmp_ocl[0] = intermediate_output_channels[i];
              tmp_icl[0] = intermediate_input_channels[i-1];
              pt->execute(tmp_icl, tmp_ocl, state);
            }
          })));
        }
      }
    }

    void join() {
      assert(launched);
      for (Thread& worker_thread : worker_thread_list) {
        worker_thread.join();
      }
      launched = false;
    }
  };

  struct StageContext
  {
    Stage* stage;
    std::vector<Task> tasks;
  };

  int rank;
  int nprocs;
  std::vector<Stage> stages;
  std::vector<StageContext> stage_context_list;

  void extract_stages(CollectionOutputEntry entry) {
    while (true) {
      Stage stage;
      stage.set_device(entry.via_transform->device);
      while (entry.via_transform->is_elementwise()) {
        stage.add_transform(entry);
        if (entry.target_collection == NULL) {
          stages.push_back(std::move(stage));
          return;
        }
        assert(entry.target_collection->outputs.size() == 1);
        entry = entry.target_collection->outputs[0];
      }
      assert(entry.via_transform->is_shuffle());
      stage.add_transform(entry);
      stages.push_back(std::move(stage));
      assert(entry.target_collection != NULL); // there must at least one element-wise operator after a shuffle
      assert(entry.target_collection->outputs.size() == 1);
      entry = entry.target_collection->outputs[0];
    }
  }

  void show_stages() {
    printf("[DEBUG] Dumping Stages Information\n");
    int num_stages = stages.size();
    printf("num_stages = %d\n", num_stages);
    for(int i=0; i<num_stages; i++) {
      printf("stage %d:\n", i);
      Stage& stage = stages[i];
      int num_workers = stage.get_workers().size();
      int num_local_workers = stage.get_local_workers().size();
      int num_transforms = stage.pt_list.size();
      printf("  num_workers = %d\n", num_workers);
      printf("  num_local_workers = %d\n", num_local_workers);
      printf("  num_transforms = %d\n", num_transforms);
      for(PTransform* pt : stage.pt_list) {
        string display_name = pt->display_name();
        printf("    %s\n", display_name.c_str());
      }
    }
  }

  void construct_tasks() {
    for (int stage_id=0; stage_id<stages.size(); stage_id++) {
      Stage& stage = stages[stage_id];
      StageContext stageContext;
      stageContext.stage = &stage;
      int num_local_workers = stage.num_local_workers();
      for (int local_worker_id = 0; local_worker_id < num_local_workers; local_worker_id++) {
        const WorkerDescriptor& local_wd = stage.get_local_workers()[local_worker_id];
        Task task(local_worker_id);
        if (local_wd.has_socket_id()) {
          task.bind_socket_id(local_wd.socket_id());
        }
        for (int i=0; i<stage.pt_list.size(); i++) {
          CollectionOutputEntry& entry = stage.pt_list[i];
          PTInstance* pti = entry.via_transform->create_instance(i);
          task.transform_instance_list.push_back(pti);
        }
        if (stage_id > 0) {
          Stage& previous_stage = stages[stage_id - 1];
          for (const WorkerDescriptor& wd : previous_stage.get_workers()) {
            int worker_rank = wd.rank();
            int worker_lid  = wd.lid();
            InputChannel* input_channel = ChannelMgr::create_input_channel(worker_rank, worker_lid);
            task.stage_input_channels.push_back(input_channel);
          }
        }
        for (int i=0; i<task.num_transforms()-1; i++) {
          LocalOutputChannel* output_channel;
          LocalInputChannel* input_channel;
          std::tie(output_channel, input_channel) = ChannelMgr::create_local_channels();
          task.intermediate_output_channels.push_back(output_channel);
          task.intermediate_input_channels.push_back(input_channel);
        }
        if (stage_id < stages.size() - 1) {
          Stage& next_stage = stages[stage_id + 1];
          for(const WorkerDescriptor& wd : next_stage.get_workers()) {
            int worker_rank = wd.rank();
            int worker_lid  = wd.lid();
            OutputChannel* output_channel = ChannelMgr::create_output_channel(worker_rank, worker_lid);
          }
        }
        stageContext.tasks.push_back(task);
      }
      stage_context_list.push_back(std::move(stageContext));
    }
  }

  void show_tasks() {
    printf("[DEBUG] Dumping Tasks Information\n");
    int num_stage_context = stage_context_list.size();
    printf("  num_stage_context = %d\n", num_stage_context);
    for(int i=0; i<num_stage_context; i++) {
      StageContext& sc = stage_context_list[i];
      printf("    %s\n", sc.stage->display_name.c_str());
      int num_tasks = sc.tasks.size();
      printf("    num_tasks = %d\n", num_tasks);
    }
  }

  void run() {
    for(StageContext& sc : stage_context_list) {
      for(Task& task : sc.tasks) {
        task.launch();
      }
    }
  }

  void join() {
    for(StageContext& sc : stage_context_list) {
      for(Task& task : sc.tasks) {
        task.join();
      }
    }
  }

  DistributedPipelineScheduler(const CollectionOutputEntryList& outputs) {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    assert(outputs.size() == 1);
    extract_stages(outputs[0]); // extract stages from this pipeline
    show_stages(); // display for debug purpose
    construct_tasks(); // construct the tasks
    show_tasks(); // display for debug purpose
  }
};

struct Pipeline {
  Pipeline() {}

  Pipeline* create() {
    return new Pipeline;
  }

  CollectionOutputEntryList outputs;

  template <typename InputT, typename OutputT>
  PCollection* apply(TaggedPTransform<InputT, OutputT>* pt) {
    PCollection<OutputT>* result = new PCollection<OutputT>;
    CollectionOutputEntry entry;
    entry.target_collection = result;
    entry.via_transform = pt;
    outputs.push_back(entry);
    return result;
  }

  void run() {
    DistributedPipelineScheduler scheduler(outputs);
    scheduler.run();
    scheduler.join();
  }
};

#include "canal_impl.h"