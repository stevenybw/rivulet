#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include <canal.h>

using namespace std;

#define CPU_GHZ 2.5

uint64_t currentTimestamp() {
  unsigned hi, lo;
  asm volatile ("CPUID\n\t"
      "RDTSC\n\t"
      "mov %%edx, %0\n\t"
      "mov %%eax, %1\n\t": "=r" (hi), "=r" (lo) : : "%rax", "%rbx", "%rcx", "%rdx");
  return ((uint64_t) hi << 32) | lo;
}

struct ParseDouble : public DoFn<string, double> 
{
  void processElement(ProcessContext& processContext) override {
    string& elem = processContext.element();
    istringstream iss(elem);
    double val;
    while(iss >> val) {
      processContext.output(std::move(val));
    }
  }
};

struct ValueToString : public DoFn<WN<double>, string>
{
  void processElement(ProcessContext& processContext) override {
    WN<double>& elem = processContext.element();
    ostringstream oss;
    oss << elem.id << " " << elem.e << "\n";
    string out_element = oss.str();
    processContext.output(std::move(out_element));
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

template <typename T>
struct SumFn : public CombineFn<T, T, T> {
  T zero_value;
  SumFn(T zero_value) : zero_value(zero_value) {}
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
  T extractOutput(T* acc) {
    return *acc;
  }
};

int main(int argc, char* argv[]) {
  assert(argc == 3);
  MPI_Init(NULL, NULL);

  char* text_path = argv[1];
  char* output_path = argv[2];

  std::unique_ptr<Pipeline> p = make_pipeline();
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(2));
  PCollection<string>* input = p->apply(TextIO::read()->from(text_path)->set_name("read from file"));
  PCollection<double>* parsed_array = input->apply(ParDo::of(ParseDouble())->set_name("parse to double"));
  PCollection<TS<double>>* ts_parsed_array = parsed_array->apply(ParDo::of(AssignTimestamp<double>())->set_name("assign timestamp"));
  PCollection<WN<double>>* wn_parsed_array = Window::FixedWindows::assign(ts_parsed_array, CPU_GHZ * 1e9);
  PCollection<WN<double>>* wn_parsed_array_shuffled = Shuffle::byWindowId(wn_parsed_array);

  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(2));
  PCollection<WN<double>>* wn_parsed_array_reduced = WindowedCombine::globally(wn_parsed_array_shuffled, SumFn<double>(0.0));
  PCollection<string>* outputs = wn_parsed_array_reduced->apply(ParDo::of(ValueToString())->set_name("to string"));
  outputs->apply(TextIO::write()->to(output_path));

  p->run();

  MPI_Finalize();
  return 0;
}
