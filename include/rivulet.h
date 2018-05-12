#include <algorithm>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>

#include <numa.h>
#include <mpi.h>

#include "common.h"
#include "channel.h"

using Thread = std::thread;
//using string = std::string;

using namespace std;

thread_local int tl_stage_id = -1;
thread_local int tl_task_id = -1;
thread_local int tl_transform_id = -1;

int g_rank;
int g_nprocs;

uint64_t g_begin_ts;

std::mutex g_mpi_lock; // protects mpi routine

uint64_t currentAbsoluteTimestamp() {
  unsigned hi, lo;
  asm volatile ("CPUID\n\t"
      "RDTSC\n\t"
      "mov %%edx, %0\n\t"
      "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo) : : "%rax", "%rbx", "%rcx", "%rdx");
  return ((uint64_t) hi << 32) | lo;
}

uint64_t currentTimestamp() {
  return currentAbsoluteTimestamp() - g_begin_ts;
}

#define MPI_DEBUG

#ifdef MPI_DEBUG

#include <signal.h>

#define assert(COND) do{if(!(COND)) {printf("ASSERTION VIOLATED, PROCESS pid = %d PAUSED\n", getpid()); while(1);}}while(0)

static void MPI_Comm_err_handler_function(MPI_Comm* comm, int* errcode, ...) {
  assert(0);
}
#define LINES do{printf("  %d> %s:%d\n", g_rank, __FUNCTION__, __LINE__);}while(0)
static void signal_handler(int sig) {
  printf("SIGNAL %d ENCOUNTERED, PROCESS pid = %d PAUSED\n", sig, getpid());
  while(true);
}
void init_debug() {
  MPI_Errhandler errhandler;
  MPI_Comm_create_errhandler(&MPI_Comm_err_handler_function,  &errhandler);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, errhandler);

  struct sigaction act;
  memset(&act, 0, sizeof(struct sigaction));
  act.sa_handler = signal_handler;
  sigaction(9, &act, NULL);
  sigaction(11, &act, NULL);
}
#else
void init_debug() {}
#endif

// Raised if the channel being closed while pulling
struct ChannelClosedException : public std::exception {
  const char* what () const throw () {
    return "ChannelClosedException";
  }
};

struct InputChannel
{
  virtual bool eos()=0;

  // TODO: ad-hoc trick here
  // is_sticky is a trick for the fact that AggregatedInputChannel may receive atoms from 
  // multiple sources, but some deserializer (like string) requires multiple atoms received 
  // continguously. Set is_sticky true to fetch from the same source of last pull
  virtual const void* pull(size_t bytes, bool is_sticky)=0;

  // poll = non-blocking pull: return NULL if not available
  virtual const void* poll(size_t bytes)=0;
};

struct OutputChannel
{
  virtual void close()=0;

  // TODO: ad-hoc trick here
  // Setting is_eager true to flush after each write. This is important if data rate
  // is too low and buffering may cause high latency.
  virtual void push(const void* data, size_t bytes, bool is_eager)=0;
};

using InputChannelList = std::vector<InputChannel*>;
using OutputChannelList = std::vector<OutputChannel*>;

// AggregatedInputChannel
//   Aggregate a list of input channels, and form a single input channel.
struct AggregatedInputChannel : public InputChannel
{
  InputChannelList input_channel_list;
  int   last_pos;
  int   num_inputs;
  int   closed_inputs;
  bool* input_valid_list;

  AggregatedInputChannel(const InputChannelList& input_channel_list) : input_channel_list(input_channel_list) {
    last_pos = 0;
    num_inputs = input_channel_list.size();
    assert(num_inputs >= 1);
    closed_inputs = 0;
    input_valid_list = new bool[num_inputs];
    for(int i=0; i<num_inputs; i++) {
      input_valid_list[i] = true;
    }
  }

  bool eos() override {
    return closed_inputs == num_inputs;
  }

  const void* pull(size_t bytes, bool sticky) override {
    if (!sticky) {
      while (true) {
        last_pos++;
        if (last_pos == num_inputs) {
          last_pos = 0;
        }
        if (input_valid_list[last_pos]) {
          const void* result = input_channel_list[last_pos]->poll(bytes);
          if (result != NULL) {
            // printf("stage %d transform %d task %d> pull from task %d for %llu bytes\n", tl_stage_id, tl_transform_id, tl_task_id, last_pos, bytes);
            return result;
          }
          if (input_channel_list[last_pos]->eos()) {
            input_valid_list[last_pos] = false;
            closed_inputs++;
          }
        }
        if (closed_inputs == num_inputs) {
          throw ChannelClosedException();
        }
      }
    } else {
      try {
        const void* result = input_channel_list[last_pos]->pull(bytes, true);
        // printf("stage %d transform %d task %d> pull from task %d for %llu bytes (sticky)\n", tl_stage_id, tl_transform_id, tl_task_id, last_pos, bytes);
        return result;
      } catch (ChannelClosedException& e) {
        assert(false);
      }
    }
  }

  const void* poll(size_t bytes) override {
    assert(false);
  }
};

const size_t MAX_PULL_BYTES = 128*1024;
const size_t LOCAL_CHANNEL_BUFFER_BYTES = 4*1024;
const size_t MAX_LOCAL_TASKS_PER_STAGE = 256;
using LocalChannelType = Channel_1<LOCAL_CHANNEL_BUFFER_BYTES>;

struct LocalInputChannel : public InputChannel
{
  LocalChannelType channel;
  char input_buffer[MAX_PULL_BYTES + LOCAL_CHANNEL_BUFFER_BYTES];
  uint64_t used_bytes, total_bytes;

  LocalInputChannel(LocalChannelType channel) : channel(channel), used_bytes(0), total_bytes(0) {}

  const void* pull(size_t bytes, bool sticky) override {
    assert(bytes <= MAX_PULL_BYTES);
    if (total_bytes - used_bytes >= bytes) {
      void* result = &input_buffer[used_bytes];
      used_bytes += bytes;
      return result;
    } else {
      memmove(&input_buffer[0], &input_buffer[used_bytes], total_bytes - used_bytes);
      total_bytes = total_bytes - used_bytes;
      used_bytes  = 0;
      while (total_bytes < bytes) {
        if (channel.eos()) {
          throw ChannelClosedException();
        }
        channel.poll([this](const char* data, size_t data_bytes) {
          memcpy(&input_buffer[total_bytes], data, data_bytes);
          total_bytes += data_bytes;
        });
      }
      used_bytes = bytes;
      return &input_buffer[0];
    }
  }

  const void* poll(size_t bytes) override {
    assert(bytes <= MAX_PULL_BYTES);

    if (total_bytes - used_bytes < bytes) {
      memmove(&input_buffer[0], &input_buffer[used_bytes], total_bytes - used_bytes);
      total_bytes = total_bytes - used_bytes;
      used_bytes  = 0;
      channel.poll([this](const char* data, size_t data_bytes) {
        memcpy(&input_buffer[total_bytes], data, data_bytes);
        total_bytes += data_bytes;
      });
    }
    if (total_bytes - used_bytes >= bytes) {
      void* result = &input_buffer[used_bytes];
      used_bytes += bytes;
      return result;
    } else {
      return NULL;
    }
  }

  bool eos() override {
    return channel.eos();
  }
};

struct LocalOutputChannel : public OutputChannel
{
  LocalChannelType channel;

  LocalOutputChannel(LocalChannelType channel) : channel(channel) { }

  void push(const void* data, size_t bytes, bool is_eager) override {
    channel.push((const char*) data, bytes);
    if (is_eager) {
      channel.flush_and_wait();
    }
  }

  void close() override {
    channel.close();
  }
};

// Maximum MPI TAG (which is 2**23 in OpenMPI)
const int MAX_MPI_TAG = (1<<22);

struct MPIOutputChannel : public OutputChannel
{
  const static size_t MPI_BUFFER_BYTES = 128*1024;
  char input_buffer[MPI_BUFFER_BYTES];
  uint64_t used_bytes;
  int to_rank;
  int identifier;
  bool is_eos;

  MPIOutputChannel(int to_rank, int identifier) : used_bytes(0), to_rank(to_rank), identifier(identifier), is_eos(false) {
    assert(identifier < MAX_MPI_TAG);
  }

  void mpi_send(char* buf, size_t bytes) {
    MPI_Request req;
    {
      std::lock_guard<std::mutex> lock(g_mpi_lock);
      MPI_Isend(buf, bytes, MPI_CHAR, to_rank, identifier, MPI_COMM_WORLD, &req);
    }
    while (true) {
      std::lock_guard<std::mutex> lock(g_mpi_lock);
      int ok;
      MPI_Status st;
      MPI_Test(&req, &ok, &st);
      if (ok) {
        break;
      }
    }
  }

  void flush() {
    if (used_bytes > 0) {
      mpi_send(input_buffer, used_bytes);
      used_bytes = 0;
    }
  }

  void push(const void* data, size_t bytes, bool is_eager) override {
    if (MPI_BUFFER_BYTES - used_bytes >= bytes) {
      memcpy(&input_buffer[used_bytes], data, bytes);
      used_bytes += bytes;
    } else {
      mpi_send(input_buffer, used_bytes);
      memcpy(&input_buffer[0], data, bytes);
      used_bytes = bytes;
    }
    if (is_eager) {
      flush();
    }
  }

  void close() override {
    flush();
    mpi_send(NULL, 0);
  }
};

struct MPIInputChannel : public InputChannel
{
  const static size_t MPI_BUFFER_BYTES = 128*1024;
  char input_buffer[MPI_BUFFER_BYTES];
  uint64_t used_bytes;
  uint64_t total_bytes;
  int from_rank;
  int identifier;
  bool is_eos;

  MPIInputChannel(int from_rank, int identifier) : used_bytes(0), total_bytes(0), from_rank(from_rank), identifier(identifier), is_eos(false) {
    assert(identifier < MAX_MPI_TAG);
  }

  const void* pull(size_t bytes, bool sticky) {
    assert(bytes <= MPI_BUFFER_BYTES);
    if (total_bytes - used_bytes >= bytes) {
      void* result = &input_buffer[used_bytes];
      used_bytes += bytes;
      return result;
    } else {
      memmove(&input_buffer[0], &input_buffer[used_bytes], total_bytes - used_bytes);
      total_bytes = total_bytes - used_bytes;
      used_bytes  = 0;
      while (true) {
        std::lock_guard<std::mutex> lock(g_mpi_lock);
        int ok;
        MPI_Status st;
        MPI_Iprobe(from_rank, identifier, MPI_COMM_WORLD, &ok, &st);
        if (ok) {
          MPI_Recv(&input_buffer[total_bytes], MPI_BUFFER_BYTES - total_bytes, MPI_CHAR, from_rank, identifier, MPI_COMM_WORLD, &st);
          int count;
          MPI_Get_count(&st, MPI_CHAR, &count);
          if (count != 0) {
            total_bytes += count;
            assert(total_bytes >= bytes);
            used_bytes = bytes;
            return &input_buffer[0];
          } else {
            is_eos = true;
            // Channel close if size = 0
            throw ChannelClosedException();
          }
        }
      }
    }
  }

  const void* poll(size_t bytes) override {
    assert(bytes <= MPI_BUFFER_BYTES);
    if (total_bytes - used_bytes >= bytes) {
      void* result = &input_buffer[used_bytes];
      used_bytes += bytes;
      return result;
    } else {
      memmove(&input_buffer[0], &input_buffer[used_bytes], total_bytes - used_bytes);
      total_bytes = total_bytes - used_bytes;
      used_bytes  = 0;
      std::lock_guard<std::mutex> lock(g_mpi_lock);
      int ok;
      MPI_Status st;
      MPI_Iprobe(from_rank, identifier, MPI_COMM_WORLD, &ok, &st);
      if (ok) {
        MPI_Recv(&input_buffer[total_bytes], MPI_BUFFER_BYTES - total_bytes, MPI_CHAR, from_rank, identifier, MPI_COMM_WORLD, &st);
        int count;
        MPI_Get_count(&st, MPI_CHAR, &count);
        if (count != 0) {
          total_bytes += count;
          assert(total_bytes >= bytes);
          used_bytes = bytes;
          return &input_buffer[0];
        } else {
          is_eos = true;
          // Channel close if size = 0
          throw ChannelClosedException();
        }
      } else {
        return NULL;
      }
    }
  }

  bool eos() override {
    return is_eos;
  }
};

struct ChannelMgr 
{
  std::map<int, std::pair<LocalChannelType, int>> local_channels;
  int rank;
  int nprocs;

  ChannelMgr() {
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  }

  InputChannel* create_input_channel(int from_rank, int from_stage, int from_lid, int to_lid) {
    assert(from_lid <= MAX_LOCAL_TASKS_PER_STAGE);
    int identifier = from_stage * MAX_LOCAL_TASKS_PER_STAGE * MAX_LOCAL_TASKS_PER_STAGE + from_lid * MAX_LOCAL_TASKS_PER_STAGE + to_lid;
    if (from_rank == rank) {
      // local input channel
      LocalChannelType channel;
      auto it = local_channels.find(identifier);
      if (it != local_channels.end()) {
        channel = it->second.first;
        it->second.second++;
      } else {
        channel.init();
        auto result = local_channels.emplace(identifier, std::make_pair(channel, 1));
        assert(result.second == true);
      }
      return new LocalInputChannel(channel);
    } else {
      // remote input channel
      return new MPIInputChannel(from_rank, identifier); // receive from the channel from worker (rank, from_stage, from_lid) that send to to_lid
    }
  }

  OutputChannel* create_output_channel(int to_rank, int from_stage, int from_lid, int to_lid) {
    assert(to_lid <= MAX_LOCAL_TASKS_PER_STAGE);
    int identifier = from_stage * MAX_LOCAL_TASKS_PER_STAGE * MAX_LOCAL_TASKS_PER_STAGE + from_lid * MAX_LOCAL_TASKS_PER_STAGE + to_lid;
    if (to_rank == rank) {
      // local output channel
      LocalChannelType channel;
      auto it = local_channels.find(identifier);
      if (it != local_channels.end()) {
        channel = it->second.first;
        it->second.second++;
      } else {
        channel.init();
        auto result = local_channels.emplace(identifier, std::make_pair(channel, 1));
        assert(result.second == true);
      }
      return new LocalOutputChannel(channel);
    } else {
      // remote output channel
      return new MPIOutputChannel(to_rank, identifier);
    }
  }

  std::tuple<LocalOutputChannel*, LocalInputChannel*> create_local_channels() {
    LocalChannelType channel;
    channel.init();
    LocalOutputChannel* out = new LocalOutputChannel(channel);
    LocalInputChannel* in = new LocalInputChannel(channel);
    return std::make_tuple(out, in);
  }

  void complete() {
    cout << "  num_channel_pairs = " << local_channels.size() << endl;
    for (auto& elem : local_channels) {
      assert(elem.second.second == 2);
    }
  }
};

void RV_Init()
{
  int required_level = MPI_THREAD_SERIALIZED;
  int provided_level;
  MPI_Init_thread(NULL, NULL, required_level, &provided_level);
  assert(provided_level >= required_level);
  MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &g_nprocs);
  g_begin_ts = currentAbsoluteTimestamp();
}

void RV_Finalize()
{
  MPI_Finalize();
}

// using ChannelList = std::vector<Channel*>;

template <typename T>
struct TS {
  TS()=default;
  TS(T&& e, uint64_t ts) : e(std::move(e)), ts(ts) {}
  T e;
  uint64_t ts;
};

template <typename T>
struct WN {
  WN()=default;
  WN(T&& e, uint64_t id) : e(std::move(e)), id(id) {}
  T e;
  uint64_t id;
};

struct WorkerDescriptor {
  int worker_rank;
  int worker_lid;
  int worker_socket_id;
  int rank() const { return worker_rank; }
  int lid() const { return worker_lid; }
  bool has_socket_id() const { return worker_socket_id>=0; }
  int socket_id() const { return worker_socket_id; }
};

// TODO: need a more accurate way to express location
struct CPUDevice {
  std::set<int> node_set; // ranks to distributed to
  std::set<int> socket_set; // socket to distributed
  int num_tasks_per_socket;

  CPUDevice* all_nodes() {
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    for(int i=0; i<nprocs; i++) {
      node_set.insert(i);
    }
    return this;
  }

  CPUDevice* all_sockets() {
    int num_sockets = numa_num_configured_nodes();
    for(int i=0; i<num_sockets; i++) {
      socket_set.insert(i);
    }
    return this;
  }

  CPUDevice* tasks_per_socket(int num_tasks_per_socket) {
    this->num_tasks_per_socket = num_tasks_per_socket;
    return this;
  }
};

struct Device {
  static CPUDevice* CPU() { return new CPUDevice; }
};

CPUDevice* g_curr_device = NULL;

void WithDevice(CPUDevice* device) {
  g_curr_device = device;
}

struct PTransform;
struct PCollectionBase;

// A PTransform instance bound with a unique id and optionally a state
struct PTInstance {
  int instance_id;
  PTransform* ptransform;
  void* state;
};

struct PTransform {
  string     name;
  CPUDevice* device;
  bool       is_eager;
  PTransform() : name(""), device(g_curr_device), is_eager(false) {}
  string display_name() { return name + " @ " + type_name(); }

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
  PCollectionBase* target_collection;
  PTransform*      via_transform;

  CollectionOutputEntry() : target_collection(NULL), via_transform(NULL) {}
};

using CollectionOutputEntryList = std::vector<CollectionOutputEntry>;

struct PCollectionBase
{
  CollectionOutputEntryList outputs;
};

template <typename T>
struct PCollection : public PCollectionBase
{
  PCollection() {}

  template <typename InputT, typename OutputT>
  PCollection<OutputT>* apply(TaggedPTransform<InputT, OutputT>* pt) {
    PCollection<OutputT>* result = new PCollection<OutputT>;
    CollectionOutputEntry entry;
    entry.target_collection = result;
    entry.via_transform = pt;
    outputs.push_back(entry);
    return result;
  }

  void set_next_transform_eager() {
    for (CollectionOutputEntry& entry : outputs) {
      entry.via_transform->is_eager = true;
    }
  }
};

// Scheduler fo a pipeline (assume a strait pipeline, no diverge), and support
// distributed environment.
struct DistributedPipelineScheduler {
  struct Stage 
  {
    Stage() : device(NULL) {}
    // @TODO add move constructor
    std::vector<PTransform*> ptransform_list;
    std::vector<WorkerDescriptor> workers;
    std::vector<WorkerDescriptor> local_workers;
    CPUDevice* device;
    void set_device(CPUDevice* device) {
      assert(device != NULL);
      this->device = device;
      int rank, nprocs;
      MPI_Comm_rank(MPI_COMM_WORLD, &rank);
      MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

      const std::set<int>& node_set = device->node_set;
      const std::set<int>& socket_set = device->socket_set;
      int num_tasks_per_socket = device->num_tasks_per_socket;
      for (int node_id : node_set) {
        int lid = 0;
        for (int socket_id : socket_set) {
          for (int i=0; i<num_tasks_per_socket; i++) {
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
    void add_ptransform(PTransform* ptransform) { ptransform_list.push_back(ptransform); }
    int num_local_workers()    { return local_workers.size(); }
    std::vector<WorkerDescriptor>& get_local_workers()    { return local_workers; }
    std::vector<WorkerDescriptor>& get_workers()    { return workers; }
  };

  struct Task 
  {
    int  stage_id;
    int  id; // unique id in a stage for this task
    int  socket_id; // socket id to bound to
    bool launched;
    std::vector<Thread> worker_thread_list; // one thread for each task transform
    Task(int stage_id, int task_id) : stage_id(stage_id), id(task_id), socket_id(-1), launched(false) {}
    Task(Task&& task) =default; // default move constructor

    void bind_socket_id(int socket_id) {
      this->socket_id = socket_id;
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
          if (socket_id > 0) {
            numa_run_on_node(socket_id);
          }
          PTInstance* instance = transform_instance_list[0];
          PTransform* ptransform = instance->ptransform;
          void* state = instance->state;
          ptransform->execute(stage_input_channels, stage_output_channels, state);
        })));
      } else {
        for(size_t i=0; i<transform_instance_list.size(); i++) {
          worker_thread_list.push_back(std::move(Thread([this, i](){
            tl_stage_id = stage_id;
            tl_transform_id = i;
            tl_task_id = id;
            if (socket_id > 0) {
              numa_run_on_node(socket_id);
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
            printf("  stage id %d, task id %d, transform id %d terminated\n", stage_id, id, i);
          })));
        }
      }
    }

    void join() {
      assert(launched);
      for (Thread& worker_thread : worker_thread_list) {
        worker_thread.join();
      }
      worker_thread_list.clear();
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
  ChannelMgr channel_mgr;

  void extract_stages(CollectionOutputEntry entry) {
    while (true) {
      Stage stage;
      stage.set_device(entry.via_transform->device);
      while (entry.via_transform->is_elementwise()) {
        stage.add_ptransform(entry.via_transform);
        if (entry.target_collection->outputs.size() == 0) {
          stages.push_back(std::move(stage));
          return;
        }
        assert(entry.target_collection->outputs.size() == 1);
        entry = entry.target_collection->outputs[0];
      }
      assert(entry.via_transform->is_shuffle());
      stage.add_ptransform(entry.via_transform);
      stages.push_back(std::move(stage));
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
      int num_transforms = stage.ptransform_list.size();
      printf("  num_workers = %d\n", num_workers);
      printf("  num_local_workers = %d\n", num_local_workers);
      printf("  num_transforms = %d\n", num_transforms);
      for(PTransform* ptransform : stage.ptransform_list) {
        string display_name = ptransform->display_name();
        printf("    %s\n", ptransform->display_name().c_str());
      }
    }
  }

  void construct_tasks() {
    for (uint64_t stage_id=0; stage_id<stages.size(); stage_id++) {
      Stage& stage = stages[stage_id];
      StageContext stageContext;
      stageContext.stage = &stage;
      int num_local_workers = stage.num_local_workers();
      for (int local_worker_id = 0; local_worker_id < num_local_workers; local_worker_id++) {
        const WorkerDescriptor& local_wd = stage.get_local_workers()[local_worker_id];
        Task task(stage_id, local_worker_id);
        if (local_wd.has_socket_id()) {
          task.bind_socket_id(local_wd.socket_id());
        }
        for (size_t i=0; i<stage.ptransform_list.size(); i++) {
          PTransform* ptransform = stage.ptransform_list[i];
          PTInstance* pti = ptransform->create_instance(local_worker_id);
          task.transform_instance_list.push_back(pti);
        }
        if (stage_id > 0) {
          Stage& previous_stage = stages[stage_id - 1];
          for (const WorkerDescriptor& wd : previous_stage.get_workers()) {
            int worker_rank = wd.rank();
            int worker_lid  = wd.lid();
            InputChannel* input_channel = channel_mgr.create_input_channel(worker_rank, stage_id-1, worker_lid, local_worker_id);
            task.stage_input_channels.push_back(input_channel);
          }
        }
        for (int i=0; i<task.num_transforms()-1; i++) {
          LocalOutputChannel* output_channel;
          LocalInputChannel* input_channel;
          std::tie(output_channel, input_channel) = channel_mgr.create_local_channels();
          task.intermediate_output_channels.push_back(output_channel);
          task.intermediate_input_channels.push_back(input_channel);
        }
        if (stage_id < stages.size() - 1) {
          Stage& next_stage = stages[stage_id + 1];
          for(const WorkerDescriptor& wd : next_stage.get_workers()) {
            int worker_rank = wd.rank();
            int worker_lid  = wd.lid();
            OutputChannel* output_channel = channel_mgr.create_output_channel(worker_rank, stage_id, local_worker_id, worker_lid);
            task.stage_output_channels.push_back(output_channel);
          }
        }
        stageContext.tasks.push_back(std::move(task));
      }
      stage_context_list.push_back(std::move(stageContext));
    }
    channel_mgr.complete();
  }

  void show_tasks() {
    printf("[DEBUG] Dumping Tasks Information\n");
    int num_stage_context = stage_context_list.size();
    printf("  num_stage_context = %d\n", num_stage_context);
    for(int i=0; i<num_stage_context; i++) {
      StageContext& sc = stage_context_list[i];
      int num_tasks = sc.tasks.size();
      printf("    num_tasks = %d\n", num_tasks);
      for(int j=0; j<num_tasks; j++) {
        Task& task = sc.tasks[j];
        int id = task.id;
        int socket_id = task.socket_id;
        int num_transforms = task.transform_instance_list.size();
        int stage_input_size = task.stage_input_channels.size();
        int stage_output_size = task.stage_output_channels.size();
        printf("    task_id=%d  socket_id=%d  num_transforms=%d  stage_input_size=%d  stage_output_size=%d\n", id, socket_id, num_transforms, stage_input_size, stage_output_size);
      }
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

  CollectionOutputEntryList outputs;

  template <typename OutputT>
  PCollection<OutputT>* apply(TaggedPTransform<PCollectionInput, OutputT>* pt) {
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

std::unique_ptr<Pipeline> make_pipeline() {
  return std::make_unique<Pipeline>();
}

#include "rivulet_impl.h"