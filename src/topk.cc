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

struct StringSplit : public DoFn<string, KV<string, long>> {
  void processElement(ProcessContext processContext) override {
    string line = std::move(processContext.element());
    istringstream iss(line);
    string token;
    while(iss >> token) {
      uint64_t ts = currentTimestamp();
      processContext.outputWithTimestamp(std::move(token), ts);
    }
  }
};

struct WordCountFormat : public DoFn<KV<string, long>, string> {
  void processElement(ProcessContext processContext) override {
    KV<string, long> item = std::move(processContext.element());
    item.key.append(ltoa(item.value));
    processContext.output(std::move(item.key));
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

  void addInput(TopkCombineFnAccumulater* rhs, const KV<string, long>& rhs) override {
    rhs->max_q.push(rhs);
    if (rhs->max_q.size() > K) {
      rhs->max_q.pop();
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

  std::list<KV<string, long>> extractOutput(TopkCombineFnAccumulater* accumulator) override {
    ostringstream oss;
    for(const KV<string, long>& kv : accumulator->max_q) {
      oss << kv.key << " " << kv.value << endl;
    }
    return oss.str();
  }
};


int main(int argc, char* argv[]) {
  assert(argc == 4);
  char* text_path = argv[1];
  char* output_path_wordcount = argv[2];
  char* output_path_topk = argv[3];
  int   topK             = atoi(argv[4]);

  Pipeline* p = Pipeline.create();
  WithDevice(Device::CPU().node(ALL).socket(ALL).cpu_per_socket(2));
  PCollection<string>* input = p->apply(TextIO::read()->from(text_path));
  PCollection<KV<string, long>>* words = input->apply(ParDo::of(StringSplit()));
  PCollection<KV<string, long>>* windowed_words = words->apply(Window::into(FixedWindows::of(CPU_GHZ * 1e9)));

  PCollection<KV<string, long>>* word_counts = windowed_words->apply(Count::perKey());
  PCollection<string>* output = word_counts->apply(ParDo::of(WordCountFormat()));
  output->apply(TextIO::write()->to(output_path_wordcount));

  PCollection<string>* topk_word_counts_output = word_counts->apply(Combine::globally(TopkCombineFn(topK)));
  output->apply(TextIO::write()->to(output_path_topk));
  p->run();

  return 0;
}
