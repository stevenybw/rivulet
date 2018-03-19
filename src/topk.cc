#include <fstream>
#include <iostream>
#include <queue>
#include <sstream>

#include <algorithm.h>
#include <assert.h>
#include <functional.h>
#include <stdio.h>
#include <stdlib.h>

using namespace std;

#define CPU_GHZ 2.5

uint64_t currentTimestamp() {
  unsigned hi, lo;
  asm volatile ("CPUID\n\t"
      "RDTSC\n\t"
      "mov %%edx, %0\n\t"
      "mov %%eax, %1\n\t": "=r" (*hi), "=r" (*lo) : : "%rax", "%rbx", "%rcx", "%rdx");
  return ((uint64_t) hi << 32) | lo;
}

struct StringSplit : public DoFn<string, TS<KV<string, long>>> {
  void processElement(ProcessContext processContext) override {
    string line = std::move(processContext.element());
    istringstream iss(line);
    string token;
    while(iss >> token) {
      uint64_t ts = currentTimestamp();
      processContext.output(make_kv(std::move(token), 1LL), ts);
    }
  }
};

template <typename T>
struct AssignTimestamp : public DoFn<T, TS<T>> {
  void processElement(ProcessContext& processContext) override {
    T elem = std::move(processContext.element());
    uint64_t ts = currentTimestamp();
    processContext.output(make_ts(std::move(elem), ts));
  }
};

struct WordCountFormat : public DoFn<WIN<KV<string, long>>, string> {
  void processElement(ProcessContext& processContext) override {
    WIN<KV<string, long>> item = std::move(processContext.element());
    ostringstream oss;
    oss << item.id << " " << item.e.key << " " << item.e.value << endl;
    processContext.output(std::move(oss.str()));
  }
};

struct TopkCombineFnAccumulater {
  using E = KV<string, long>;
  struct Compare {
    bool operator<(const E& lhs, const E& rhs) {
      return lhs.value < rhs.value;
    }
  };
  using MaxQ = std::priority_queue<E, std::vector<E>, Compare>;

  MaxQ max_q;
};

struct TopkCombineFn : public CombineFn<KV<string, long>, TopkCombineFnAccumulater, string> {
  long K;

  TopkCombineFn(long K) {
    this->K = K;
  }

  TopkCombineFnAccumulater* createAccumulator() override {
    return new TopkCombineFnAccumulater;
  }

  void addInput(TopkCombineFnAccumulater* lhs, const KV<string, long>& rhs) override {
    lhs->max_q.push(rhs);
    if (lhs->max_q.size() > K) {
      lhs->max_q.pop();
    }
  }

  TopkCombineFnAccumulater* mergeAccumulators(const std::list<TopkCombineFnAccumulater*>& accumulators) override {
    TopkCombineFnAccumulater* result_acc = createAccumulator();
    for(TopkCombineFnAccumulater* acc : accumulators) {
      for(const KV<string, long>& kv : acc->max_q) {
        result_acc->max_q.push(kv);
        if (result_acc->max_q.size() > K) {
          result_acc->max_q.pop();
        }
      }
    }
    return result_acc;
  }

  string extractOutput(TopkCombineFnAccumulater* accumulator) override {
    ostringstream oss;
    for(const KV<string, long>& kv : accumulator->max_q) {
      oss << kv.key << " " << kv.value << endl;
    }
    return oss.str();
  }
};

template <typename T>
struct SumFn : public CombineFn<T, typename T zero_value, T, T> {
  SumFn() {}
  T* createAccumulator() override {
    return new T(zero_value);
  }
  void resetAccumulator(T* acc) {
    (*acc) = zero_value;
  }
  void addInput(T* lhs, const T& rhs) override {
    (*lhs) += rhs;
  }
  T* mergeAccumulators(const std::list<T*>& accumulators) override {
    T* result_acc = createAccumulator();
    for(T* acc : accumulators) {
      (*result_acc) += (*acc);
    }
    return result_acc;
  }
};


int main(int argc, char* argv[]) {
  assert(argc == 4);
  char* text_path = argv[1];
  char* output_path_wordcount = argv[2];
  char* output_path_topk = argv[3];
  int   topK             = atoi(argv[4]);

  Pipeline* p = Pipeline.create();
  // scatter for better IO performance
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->task_per_socket(2));
  PCollection<string>* input = p->apply(TextIO::read()->from(text_path)->set_name("read from file"));
  PCollection<KV<string, long>>* words = input->apply(ParDo::of(StringSplit())->set_name("split lines"));
  PCollection<TS<KV<string, long>>>* ts_words = words->apply(ParDo::of(AssignTimestamp<KV<string, long>>())->set_name("assign timestamp"));
  PCollection<WN<KV<string, long>>>* windowed_words = Window::FixedWindows::assign(ts_words, CPU_GHZ * 1e9); // ts_words->apply(Window::FixedWindows::of(CPU_GHZ * 1e9));
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->task_per_socket(2));
  PCollection<WN<KV<string, long>>>* windowed_word_counts = windowed_words->apply(WindowedCombine::perKey());
  PCollection<string>* output = word_counts->apply(ParDo::of(WinWordCountFormat()));
  output->apply(TextIO::write()->to(output_path_wordcount));
  
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->task_per_socket(2));
  PCollection<string>* topk_word_counts_output = word_counts->apply(Combine::globally(TopkCombineFn(topK)));
  output->apply(TextIO::write()->to(output_path_topk));
  p->run();

  return 0;
}
