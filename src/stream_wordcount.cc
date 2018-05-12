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

  init_debug();

  assert(argc == 3);

  string output_path = argv[1];
  string read_from = argv[2];

  std::unique_ptr<Pipeline> p = make_pipeline();
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(1));
  PCollection<string>* words;
  if (read_from == "__random__") {
    words = p->apply(Generator::of(GenerateEnglishWord(dict)));
  } else {
    words = p->apply(TextIO::read()->from(read_from.c_str())->set_name("read from file"));
  }
  PCollection<TS<string>>* words_ts = words->apply(ParDo::of(AssignTimestamp<string>())->set_name("assign timestamp"));
  PCollection<WN<string>>* words_wn = Window::FixedWindows::assign(words_ts, CPU_GHZ * 5e8);
  PCollection<WN<string>>* words_wn_shuffled = Shuffle::byWindowId(words_wn);

  PCollection<WN<map<string,int>>>* wordcounts = WindowedCombine::globally(words_wn_shuffled, WordCountTopKFn(10));
  PCollection<string>* outputs = wordcounts->apply(ParDo::of(WordCountToString())->set_name("to string"));
  outputs->apply(TextIO::write()->to(output_path.c_str()));

  words_wn_shuffled->set_next_transform_eager();
  wordcounts->set_next_transform_eager();
  outputs->set_next_transform_eager();

  p->run();

  RV_Finalize();
  return 0;
}