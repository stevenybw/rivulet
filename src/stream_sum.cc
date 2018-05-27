#include <algorithm>
#include <fstream>
#include <functional>
#include <iostream>
#include <queue>
#include <sstream>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "rivulet.h"

using namespace std;

#define CPU_GHZ 2.5

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

struct DoubleToString : public DoFn<TS<double>, string>
{
  void processElement(ProcessContext& processContext) override {
    auto& elem = processContext.element();
    ostringstream oss;
    oss << "ts = " << elem.ts << "  e = " << elem.e;
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

struct GenerateRandomDouble : public GenFn<double>
{
  const static uint64_t a = 6364136223846793005LL;
  const static uint64_t c = 1442695040888963407LL;
  double min;
  double max;
  uint64_t mmix_state;
  GenerateRandomDouble(const GenerateRandomDouble& rhs)=default;
  GenerateRandomDouble(double min, double max) : min(min), max(max), mmix_state(c) {}
  void set_instance_id(int instance_id) override {
    mmix_state += instance_id;
  }
  double generateElement() override {
    mmix_state = a * mmix_state + c;
    return (max-min) * (1.0 * (mmix_state&0xFFFFFFFFLL) / 0xFFFFFFFFLL) + min;
  }
};

using SumCountPair = std::pair<double, uint64_t>;

struct SumFn : public CombineFn<double, SumCountPair, SumCountPair> {
  SumFn() {}
  SumCountPair* createAccumulator() override {
    SumCountPair* acc = new SumCountPair;
    acc->first = 0.0;
    acc->second = 0;
  }
  void resetAccumulator(SumCountPair* acc) {
    acc->first = 0.0;
    acc->second = 0;
  }
  void addInput(SumCountPair* lhs, const double& rhs) override {
    lhs->first += rhs;
    lhs->second ++;
  }
  SumCountPair* mergeAccumulators(const std::list<SumCountPair*>& accumulators) override {
    SumCountPair* result_acc = createAccumulator();
    for(SumCountPair* acc : accumulators) {
      result_acc->first += acc->first;
      result_acc->second += acc->second;
    }
    return result_acc;
  }
  SumCountPair extractOutput(SumCountPair* acc) {
    return *acc;
  }
};

struct ValueToString : public DoFn<WN<SumCountPair>, string>
{
  void processElement(ProcessContext& processContext) override {
    WN<SumCountPair>& elem = processContext.element();
    ostringstream oss;
    oss << "wnid: " << elem.id << "   sum: " << elem.e.first << "  count: " << elem.e.second << "   average: " << (1.0*elem.e.first/elem.e.second);
    string out_element = oss.str();
    processContext.output(std::move(out_element));
  }
};

int main(int argc, char* argv[]) {
  assert(argc >= 3);
  RV_Init();

  init_debug();

  string text_path = argv[1];
  string output_path = argv[2];

  std::unique_ptr<Pipeline> p = make_pipeline();
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(2));
  PCollection<double>* parsed_array = NULL;
  if (text_path == "__random__") {
    assert(argc == 4);
    double mu = atof(argv[3]);
    parsed_array = p->apply(Generator::of(GenerateRandomDouble(0.0, 2.0 * mu)));
  } else {
    FileSystemWatch* fs_watch = new FileSystemWatch;
    fs_watch->add_watch(text_path);
    PCollection<string>* input = p->apply(TextIO::readTextFromWatch(fs_watch));
    parsed_array = input->apply(ParDo::of(ParseDouble()));
  }
  PCollection<TS<double>>* ts_parsed_array = parsed_array->apply(ParDo::of(AssignTimestamp<double>())->set_name("assign timestamp"));
  PCollection<WN<double>>* wn_parsed_array = Window::FixedWindows::assign(ts_parsed_array, CPU_GHZ * 5e8);
  // measure_ops(wn_parsed_array);

  PCollection<WN<double>>* wn_parsed_array_shuffled = Shuffle::byWindowId(wn_parsed_array);
  WithDevice(Device::CPU()->all_nodes()->all_sockets()->tasks_per_socket(1));
  PCollection<WN<SumCountPair>>* wn_parsed_array_reduced = WindowedCombine::globally(wn_parsed_array_shuffled, SumFn());
  PCollection<string>* outputs = wn_parsed_array_reduced->apply(ParDo::of(ValueToString())->set_name("to string"));
  outputs->apply(TextIO::write()->to(output_path.c_str()));
  wn_parsed_array_shuffled->set_next_transform_eager();
  wn_parsed_array_reduced->set_next_transform_eager();
  outputs->set_next_transform_eager();

  p->run();

  RV_Finalize();
  return 0;
}
