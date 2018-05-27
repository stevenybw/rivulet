#include <fstream>
#include <iostream>
#include <map>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <cassert>
#include <cstdio>
#include <cstdlib>

#include "rivulet.h"

using namespace std;

#define CPU_GHZ 2.5

struct Dictionary
{
  vector<string> dict;

  void load(string from_path) {
    ifstream fin(from_path);
    if (!fin) {
      printf("Unable to open dictionary file %s\n", from_path.c_str());
      assert(false);
    }
    string word;
    while (fin >> word) {
      dict.push_back(std::move(word));
    }
    printf("[Dict] loaded %zu words from %s\n", dict.size(), from_path.c_str());
  }

  void load() {
    const char* words[] = {"yuppies","yuppy","yups","z","zanier","zanies","zaniest","zaniness","zany","zap","zapped","zapping","zaps","zeal","zealot","zealots","zealous","zealously","zealousness","zebra","zebras","zebu","zebus","zed","zeds","zenith","zeniths","zephyr","zephyrs","zeppelin","zeppelins","zero","zeroed","zeroes","zeroing","zeros","zest","zestful","zestfully","zests","zeta","zigzag"};
    for(const char* word : words) {
      dict.push_back(word);
    }
    printf("[Dict] loaded %zu words\n", dict.size());
  }
};

template <typename T>
struct AssignTimestamp : public DoFn<T, TS<T>> 
{
  void processElement(typename AssignTimestamp::ProcessContext& processContext) override {
    T& elem = processContext.element();
    uint64_t ts = currentTimestamp();
    processContext.output(TS<T>(std::move(elem), ts));
  }
};

struct StringSplit : public DoFn<string, string> 
{
  constexpr static char* delimer = " ";

  void processElement(typename StringSplit::ProcessContext& processContext) override {
    string& elem = processContext.element();
    char buf[elem.size()+1];
    memcpy(buf, elem.c_str(), elem.size());
    buf[elem.size()] = '\0';
    char* saved_ptr = NULL;
    char* token = strtok_r(buf, delimer, &saved_ptr);
    while (token != NULL) {
      processContext.output(std::string(token));
      token = strtok_r(NULL, delimer, &saved_ptr);
    }
  }
};

struct WordCountToString : public DoFn<WN<map<string, int>>, string> 
{
  void processElement(ProcessContext& processContext) override {
    auto& elem = processContext.element();
    ostringstream oss;
    oss << "id = " << elem.id << "\n";
    for (auto& kv : elem.e) {
      const string& word = kv.first;
      int count = kv.second;
      oss << "  " << word << ": " << count << "\n";
    }
    processContext.output(std::move(oss.str()));
  }
};

struct GenerateEnglishWord : public GenFn<string>
{
  Dictionary& dict;
  size_t num_words;
  default_random_engine rng;
  uniform_int_distribution<size_t> dist;
  GenerateEnglishWord(const GenerateEnglishWord& rhs)=default;
  GenerateEnglishWord(Dictionary& dict) : dict(dict), num_words(dict.dict.size()), dist(0, num_words-1) { }
  void set_instance_id(int instance_id) override {
    rng.seed(instance_id);
  }
  string generateElement() override {
    return dict.dict[dist(rng)];
  }
};

template <typename InputT>
struct MeasureOpsTransform : public BasePTransform<InputT, PCollectionOutput>
{
  void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
    assert(inputs.size() == 1);
    InputChannel* in = inputs[0];
    uint64_t ts = currentTimeUs();
    uint64_t op = 0;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    while (true) {
      InputT in_element = Serdes<InputT>::stream_deserialize(in);
      op++;
      uint64_t ts_now = currentTimeUs();
      if (ts_now - ts > 1e6) {
        double kops = 1.0e3 * op / (ts_now - ts);
        printf("%d.%d> performance = %lf KOPS\n", rank, tl_task_id, kops);
        ts = currentTimeUs();
        op = 0;
      }
    }
  }
};

template <typename T>
void measure_ops(PCollection<T>* pc) {
  pc->apply(new MeasureOpsTransform<T>());
}

/*! \brief Accumulator for accumulating word count, and emit the result for each specified duration
 *
 */
struct WordCountAccumulateTopKPTransform : public BasePTransform<string, pair<string, size_t>>
{
  int K;
  uint64_t duration_us;

  struct State {
    std::map<string, size_t> wordcount;
  };

  void* new_state() override {
    return new State;
  }

  /*! \brief Set is_shuffle to forward its output into topk directly
   *
   */
  bool is_shuffle() override { return true; }

  WordCountAccumulateTopKPTransform(int K, uint64_t duration_us) : K(K), duration_us(duration_us) {}

  void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    assert(state);
    State* typed_state = (State*) state;
    std::map<string, size_t>& wordcount = typed_state->wordcount;
    assert(inputs.size() == 1);
    assert(outputs.size() == 1);
    InputChannel* in = inputs[0];
    OutputChannel* out = outputs[0];
    uint64_t ts = currentTimeUs();
    uint64_t op = 0;
    while (true) {
      string in_element = Serdes<string>::stream_deserialize(in);
      wordcount[in_element]++;
      op++;
      uint64_t ts_now = currentTimeUs();
      if (ts_now - ts > duration_us) {
        double kops = 1.0e3 * op / (ts_now - ts);
        printf("%d.%d> Accumulate performance = %lf KOPS\n", rank, tl_task_id, kops);
        for(const pair<string, size_t>& entry : wordcount) {
          Serdes<pair<string, size_t>>::stream_serialize(out, entry, this->is_eager);
          // printf("%s %zu\n", entry.first.c_str(), entry.second);
        }
        wordcount.clear();
        ts = currentTimeUs();
        op = 0;
      }
    }
  }
};

struct WordCountTopKPTransform : public BaseCombinePTransform<pair<string, size_t>, PCollectionOutput>
{
  //std::mutex mu;
  std::map<string, size_t> wordcount;
  int K;
  uint64_t duration_us;

  WordCountTopKPTransform(int K, uint64_t duration_us) : K(K), duration_us(duration_us) {}

  vector<pair<string, size_t>> topk() {
    using Pair = std::pair<size_t, string>;
    using Heap = priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;
    Heap heap;
    for(auto it=wordcount.begin(); it!=wordcount.end(); it++) {
      heap.emplace(std::make_pair(it->second, it->first));
      if (heap.size() > K) {
        heap.pop();
      }
    }
    size_t num_result = heap.size();
    vector<pair<string, size_t>> result;
    while (! heap.empty()) {
      const Pair& top = heap.top();
      result.emplace_back(std::make_pair(top.second, top.first));
      heap.pop();
    }
    return std::move(result);
  }

  void execute_combine(InputChannel* in_channel, OutputChannel* out_channel, void* state) override {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    InputChannel* in = in_channel;
    OutputChannel* out = out_channel;
    uint64_t ts = currentTimeUs();
    uint64_t op = 0;
    while (true) {
      pair<string, size_t> in_element = Serdes<pair<string, size_t>>::stream_deserialize(in);
      //std::lock_guard<std::mutex> lk(mu);
      wordcount[in_element.first] += in_element.second;
      op++;
      uint64_t ts_now = currentTimeUs();
      if (ts_now - ts > duration_us) {
        double kops = 1.0e3 * op / (ts_now - ts);
        ts = currentTimeUs();
        op = 0;
        vector<pair<string, size_t>> topk_result = topk();
        printf("TOPK RESULT\n");
        for (auto& p : topk_result) {
          printf("%20s %10zu\n", p.first.c_str(), p.second);
        }
      }
    }
  }
};

/*! \brief Combine Function for Word Count + TopK
 *
 */
struct WordCountTopKFn : public CombineFn<string, map<string, int>, map<string, int>>
{
  int K;
  WordCountTopKFn(int K) : K(K) {}

  map<string, int>* createAccumulator() override {
    return new map<string, int>();
  }

  void resetAccumulator(map<string, int>* acc) override {
    acc->clear();
  }

  void addInput(map<string, int>* acc, const string& rhs) override {
    (*acc)[rhs]++;
  }

  // TODO: Who free those allocated objects?
  map<string, int>* mergeAccumulators(const std::list<map<string, int>*>& accumulators) override {
    map<string, int>* result_acc = createAccumulator();
    for (auto acc : accumulators) {
      for(auto& kv : (*acc)) {
        const string& word = kv.first;
        int count = kv.second;
        (*result_acc)[word] += count;
      }
    }
    return result_acc;
  }

  map<string, int> extractOutput(map<string, int>* acc) {
    using Pair = std::pair<int, string>;
    using Heap = priority_queue<Pair, std::vector<Pair>, std::greater<Pair>>;
    Heap heap;
    for(auto it=acc->begin(); it!=acc->end(); it++) {
      heap.emplace(std::make_pair(it->second, it->first));
      if (heap.size() > K) {
        heap.pop();
      }
    }

    std::map<string, int> result;
    while (! heap.empty()) {
      const Pair& top = heap.top();
      result.emplace(std::make_pair(top.second, top.first));
      heap.pop();
    }
    return std::move(result);
  }
};

int main(int argc, char* argv[]) {
  Dictionary dict;
  dict.load();
  RV_Init();

  Configuration env_config;
  ExecutionContext ctx(env_config.nvm_off_cache_pool_dir, env_config.nvm_off_cache_pool_dir, env_config.nvm_on_cahce_pool_dir, MPI_COMM_WORLD);
  Driver* driver = new Driver(ctx);

  init_debug();

  assert(argc == 3);

  string read_from = argv[1];
  string output_path = argv[2];

  std::unique_ptr<Pipeline> p = make_pipeline();
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(1));
  PCollection<string>* words;
  if (read_from == "__random__") {
    words = p->apply(Generator::of(GenerateEnglishWord(dict)));
  } else {
    FileSystemWatch* fs_watch = new FileSystemWatch;
    fs_watch->add_watch(read_from);
    PCollection<string>* lines = p->apply(TextIO::readTextFromWatch(fs_watch));
    words = lines->apply(ParDo::of(StringSplit())->set_name("string split"));
  }
  PCollection<pair<string, size_t>>* wordcounts = words->apply(new WordCountAccumulateTopKPTransform(10, 1e6));

  WithDevice(Device::CPU()->one_node()->one_socket()->tasks_per_socket(1));
  wordcounts->apply(new WordCountTopKPTransform(10, 1e6));
  // measure_ops(topk);
//  PCollection<TS<string>>* words_ts = words->apply(ParDo::of(AssignTimestamp<string>())->set_name("assign timestamp"));
//  PCollection<WN<string>>* words_wn = Window::FixedWindows::assign(words_ts, CPU_GHZ * 5e8);
//  PCollection<WN<string>>* words_wn_shuffled = Shuffle::byWindowId(words_wn);
//
//  PCollection<WN<map<string,int>>>* wordcounts = WindowedCombine::globally(words_wn_shuffled, WordCountTopKFn(10));
//  PCollection<string>* outputs = wordcounts->apply(ParDo::of(WordCountToString())->set_name("to string"));
//  outputs->apply(TextIO::writeAsGarray<256>(driver)->to(output_path.c_str()));

  words->set_next_transform_eager();
//  wordcounts->set_next_transform_eager();
//  outputs->set_next_transform_eager();

  p->run();

  RV_Finalize();
  return 0;
}