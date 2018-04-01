struct __attribute__((packed)) Header {
  int    type;
  size_t bytes;
};

enum {
  TYPE_STRING = 1,
  TYPE_TS     = 2,
  TYPE_WN     = 3
};

// Serializers and deserializers
// TODO inconsistent interface: serialize, stream_deserialize
template <typename T>
struct Serdes
{
  // Serialize a C++ object into a given ostringstream
  static void serialize(ostringstream& oss, T& elem) {
    cerr << "[Error] Not Implemented Serdes" << endl;
    assert(false);
  }

  // Deserialize a C++ object from an input channel
  static T stream_deserialize(InputChannel* in, bool sticky = false) {
    cerr << "[Error] Not Implemented Serdes" << endl;
    assert(false);
  }

  // Serialize a C++ object into an output channel
  static void stream_serialize(OutputChannel* out, T& elem) {
    cerr << "[Error] Not Implemented Serdes"
    assert(false);
  }
};

// Match TS type via partial specification
template <typename T>
struct Serdes<TS<T>>
{
  static void serialize(ostringstream& oss, TS<T>& elem) {
    Header header = {TYPE_TS, elem.ts};
    oss.write((const char*) &header, sizeof(header));
    Serdes<T>::serialize(oss, elem.e);
  }
  static TS<T> stream_deserialize(InputChannel* in, bool sticky = false) {
    Header* buf = (Header*) in->pull(sizeof(Header), sticky);
    assert(buf->type == TYPE_TS);
    size_t ts = buf->bytes;
    T elem = Serdes<T>::stream_deserialize(in, true);
    return TS<T>(std::move(elem), ts);
  }
  static void stream_serialize(OutputChannel* out, TS<T>& elem, bool is_eager) {
    ostringstream oss;
    serialize(oss, elem);
    string result = oss.str();
    out->push(result.c_str(), result.size(), is_eager);
  }
};

template <typename T>
struct Serdes<WN<T>>
{
  static void serialize(ostringstream& oss, WN<T>& elem) {
    Header header = {TYPE_WN, elem.id};
    oss.write((const char*) &header, sizeof(header));
    Serdes<T>::serialize(oss, elem.e);
  }

  static WN<T> stream_deserialize(InputChannel* in, bool sticky = false) {
    Header* buf = (Header*) in->pull(sizeof(Header), sticky);
    assert(buf->type == TYPE_WN);
    size_t id = buf->bytes;
    T elem = Serdes<T>::stream_deserialize(in, true);
    return WN<T>(std::move(elem), id);
  }

  static void stream_serialize(OutputChannel* out, WN<T>& elem, bool is_eager) {
    ostringstream oss;
    serialize(oss, elem);
    string result = oss.str();
    out->push(result.c_str(), result.size(), is_eager);
  }
};

template <>
struct Serdes<string>
{
  static void serialize(ostringstream& oss, string& elem) {
    Header header = {TYPE_STRING, elem.size()};
    oss.write((const char*) &header, sizeof(header));
    oss.write(elem.c_str(), elem.size());
  }

  static string stream_deserialize(InputChannel* in, bool sticky = false) {
    Header* buf = (Header*) in->pull(sizeof(Header), sticky);
    assert(buf->type == TYPE_STRING);
    size_t bytes = buf->bytes;
    char* str = (char*) in->pull(bytes, true);
    return string(str, bytes);
  }

  static void stream_serialize(OutputChannel* out, string& elem, bool is_eager) {
    ostringstream oss;
    serialize(oss, elem);
    string result = oss.str();
    out->push(result.c_str(), result.size(), is_eager);
  }
};

template <>
struct Serdes<double>
{
  static void serialize(ostringstream& oss, double& elem) {
    oss.write((const char*) &elem, sizeof(elem));
  }

  static double stream_deserialize(InputChannel* in, bool sticky = false) {
    double* ptr = (double*) in->pull(sizeof(double), sticky);
    return *ptr;
  }

  static void stream_serialize(OutputChannel* out, double& elem, bool is_eager) {
    ostringstream oss;
    serialize(oss, elem);
    string result = oss.str();
    out->push(result.c_str(), result.size(), is_eager);
  }
};

template <typename InputT, typename OutputT>
struct DoFn {
  typedef InputT InputType;
  typedef OutputT OutputType;

  struct ProcessContext {
    InputT&        _element;
    OutputChannel* _output_channel;
    int            _is_eager;
    ProcessContext(InputT& element, OutputChannel* output_channel, bool is_eager) : _element(element), _output_channel(output_channel), _is_eager(is_eager) {}
    
    InputT& element() { return _element; }

    void output(OutputT&& output_element) {
      Serdes<OutputT>::stream_serialize(_output_channel, output_element, this->_is_eager);
    }
  };
  virtual void processElement(ProcessContext& processContext)=0;
};

// Used in Generator::of
template <typename OutputT>
struct GenFn {
  typedef OutputT OutputType;

  // Generate an object
  virtual OutputT generateElement()=0;

  // Set the instance id of this generate functor
  virtual void set_instance_id(int instance_id)=0;
};

struct ParDo {
  template <typename DoFnType>
  struct ParDoTransform : public TaggedPTransform<typename DoFnType::InputType, typename DoFnType::OutputType>
  {
    using InputT = typename DoFnType::InputType;
    using OutputT = typename DoFnType::OutputType;

    DoFnType do_fn;
    ParDoTransform(const DoFnType& do_fn) : do_fn(do_fn) {}

    ParDoTransform<DoFnType>* set_name(const string& name) { this->name = name; return this; }

    string type_name() override { return "ParDoTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      instance->state = NULL;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(inputs.size() == 1);
      InputChannel* in = inputs[0];
      OutputChannel* out = outputs[0];
      try {
        while(true) {
          InputT in_element = Serdes<InputT>::stream_deserialize(in);
          typename DoFnType::ProcessContext pc(in_element, out, this->is_eager);
          do_fn.processElement(pc);
        }
      } catch (ChannelClosedException& e) {
        assert(in->eos());
      }
      out->close();
    }
  };

  template <typename DoFnType>
  static ParDoTransform<DoFnType>* of(const DoFnType& do_fn) {
    return new ParDoTransform<DoFnType>(do_fn);
  }
};

struct Generator
{
  template <typename GenFnType>
  struct GeneratorTransform : public TaggedPTransform<PCollectionInput, typename GenFnType::OutputType>
  {
    using OutputT = typename GenFnType::OutputType;

    GenFnType gen_fn;
    GeneratorTransform(const GenFnType& gen_fn) : gen_fn(gen_fn) {}

    string type_name() override { return "GeneratorTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      GenFnType* curr_gen_fn = new GenFnType(gen_fn);
      curr_gen_fn->set_instance_id(instance_id);
      instance->state = curr_gen_fn;

      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      GenFnType* curr_gen_fn = (GenFnType*) state;
      assert(inputs.size() == 0);
      assert(outputs.size() == 1);
      OutputChannel* out = outputs[0];
      try {
        while(true) {
          OutputT out_element = curr_gen_fn->generateElement();
          Serdes<OutputT>::stream_serialize(out, out_element, this->is_eager);
        }
      } catch (ChannelClosedException& e) {
      }
      out->close();
    }
  };

  template <typename GenFnType>
  static GeneratorTransform<GenFnType>* of(const GenFnType& gen_fn) {
    return new GeneratorTransform<GenFnType>(gen_fn);
  }
};

struct Shuffle
{
  template <typename T>
  struct ShuffleByWindowIdTransform : public TaggedPTransform<WN<T>,WN<T>>
  {
    using InputT = WN<T>;
    using OutputT = WN<T>;
    string type_name() override { return "ShuffleByWindowIdTransform"; }
    bool is_elementwise() override { return false; }
    bool is_shuffle() override { return true; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      instance->state = instance;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(inputs.size() == 1);
      InputChannel* in = inputs[0];
      int num_outputs = outputs.size();
      try {
        while (true) {
          InputT in_element = Serdes<InputT>::stream_deserialize(in);
          int dest = (in_element.id) % num_outputs;
          Serdes<InputT>::stream_serialize(outputs[dest], in_element, this->is_eager);
          // printf("stage %d transform %d(Shuffle) task %d> push to %d\n", tl_stage_id, tl_transform_id, tl_task_id, dest);
        }
      } catch (ChannelClosedException& e) {
        assert(in->eos());
      }
      for (int i=0; i<num_outputs; i++) {
        outputs[i]->close();
      }
    }
  };

  template <typename T>
  static PCollection<WN<T>>* byWindowId(PCollection<WN<T>>* pcollection) {
    return pcollection->apply(new ShuffleByWindowIdTransform<T>());
  }
};

struct TextIO {
  struct ReadTransform : public TaggedPTransform<PCollectionInput, string> 
  {
    struct State {
      std::ifstream fin;
      bool last_line_valid;
      string last_line;

      State(const string& path) : fin(path), last_line_valid(false) { }
    };

    string path;

    ReadTransform* from(const string& read_path) {
      path = read_path;
      return this;
    }

    ReadTransform* set_name(const string& name) { this->name = name; return this; }

    string type_name() override { return "ReadTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      string read_path = path + "." + std::to_string(instance_id);
      cout << instance_id << ": " << " read from " << read_path << endl;
      State* state = new State(read_path);
      if (!(state->fin)) {
        cerr << "failed to open file " << read_path << endl;
        assert(false);
      }
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      instance->state = (void*) state;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(inputs.size() == 0);
      assert(outputs.size() == 1);
      OutputChannel* out = outputs[0];
      State* st = (State*) state;
      string line;
      std::ifstream& fin = st->fin;
      while (std::getline(fin, line)) {
        Serdes<string>::stream_serialize(out, line, this->is_eager);
      }
      assert(!fin);
      out->close();
    }
  };

  struct WriteTransform : public TaggedPTransform<string, PCollectionOutput>
  {
    string write_path_prefix;
    WriteTransform* to(const string& write_path_prefix) {
      this->write_path_prefix = write_path_prefix;
      return this;
    }

    WriteTransform* set_name(const string& name) { this->name = name; return this; }

    string type_name() override { return "WriteTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      instance->state = instance;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(inputs.size() == 1);
      assert(outputs.size() == 0);
      int instance_id = ((PTInstance*)state)->instance_id;
      InputChannel* in = inputs[0];
      string write_path = write_path_prefix + "." + std::to_string(instance_id);
      std::ofstream fout(write_path);
      if (!fout) {
        cerr << "failed to open file " << write_path << endl;
        assert(false);
      }
      printf("%d: written to %s\n", instance_id, write_path.c_str());
      try {
        while (true) {
          string in_element = Serdes<string>::stream_deserialize(in);
          printf("stage %d transform %d(Combine) task %d> write %s\n", tl_stage_id, tl_transform_id, tl_task_id, in_element.c_str());
          fout << in_element << endl;
        }
      } catch (ChannelClosedException& ex) {
        fout.close();
      }
    }
  };

  static ReadTransform* read() {
    return new ReadTransform;
  }

  static WriteTransform* write() {
    return new WriteTransform;
  }
};

struct Window 
{
  struct FixedWindows 
  {
    template <typename T>
    struct FixedWindowsTransform : public TaggedPTransform<TS<T>, WN<T>> 
    {
      uint64_t window_size;
      FixedWindowsTransform(uint64_t window_size) : window_size(window_size) {}

      string type_name() override { return "FixedWindowsTransform"; }
      bool is_elementwise() override { return true; }
      bool is_shuffle() override { return false; }

      PTInstance* create_instance(int instance_id) override {
        PTInstance* pti = new PTInstance;
        pti->instance_id = instance_id;
        pti->ptransform  = this;
        pti->state       = NULL;
        return pti;
      }

      void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
        assert(inputs.size() == 1);
        InputChannel* in = inputs[0];
        OutputChannel* out = outputs[0];
        try {
          while (true) {
            TS<T> in_element = Serdes<TS<T>>::stream_deserialize(in);
            uint64_t ts = in_element.ts;
            WN<T> out_element(std::move(in_element.e), ts / window_size);
            Serdes<WN<T>>::stream_serialize(out, out_element, this->is_eager);
          }
        } catch (ChannelClosedException& e) {
          assert(in->eos());
        }
        out->close();
      }
    };

    template <typename T>
    static PCollection<WN<T>>* assign(PCollection<TS<T>>* pcollection, uint64_t window_size) {
      return pcollection->apply(new FixedWindowsTransform<T>(window_size));
    }
  };
};

template <typename InT, typename AccT, typename OutT>
struct CombineFn {
  typedef InT InputType;
  typedef AccT AccumulatorType;
  typedef OutT OutputType;

  virtual AccT* createAccumulator()=0;
  virtual void  resetAccumulator(AccT* acc)=0;
  virtual void  addInput(AccT* acc, const InT& elem)=0;
  virtual AccT* mergeAccumulators(const std::list<AccT*>& accumulators)=0;
  virtual OutT  extractOutput(AccT* accumulator)=0;
};

// Aggregate each window and emit the result into subsequent transforms. This is 
// N->1  transform
struct WindowedCombine
{
  template <typename CombineFnType, typename InT, typename OutT>
  struct WindowedCombineTransform : public TaggedPTransform<WN<InT>, WN<OutT>>
  {
    // num_windows must be a prime number...
    const static uint64_t num_windows = 3;
    CombineFnType combine_fn;

    WindowedCombineTransform(const CombineFnType& combine_fn) : combine_fn(combine_fn) {}

    string type_name() override { return "WindowedCombineTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* pti = new PTInstance;
      pti->instance_id = instance_id;
      pti->ptransform  = this;
      pti->state       = pti;
      return pti;
    }

    void execute (const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      PTInstance* pti = (PTInstance*) state;
      int instance_id = pti->instance_id;

      typename CombineFnType::AccumulatorType accumulator_list[num_windows];
      uint64_t wnid_list[num_windows];
      for(int i=0; i<num_windows; i++) {
        combine_fn.resetAccumulator(&accumulator_list[i]);
        wnid_list[i] = (uint64_t) -1;
      }

      assert(outputs.size() == 1);
      OutputChannel* out = outputs[0];
      AggregatedInputChannel aggregated_in(inputs);
      try {
        while(true) {
          // fprintf(stderr, "%d> try receive\n", instance_id);
          WN<InT> in_element = Serdes<WN<InT>>::stream_deserialize(&aggregated_in);
          // fprintf(stderr, "%d> received wnid = %llu\n", instance_id, in_element.id);
          uint64_t wnid = in_element.id;
          uint64_t idx  = wnid % num_windows;
          if (wnid_list[idx] == wnid) {
            combine_fn.addInput(&accumulator_list[idx], in_element.e);
          } else {
            if (wnid_list[idx] != (uint64_t)-1) {
              WN<OutT> out_element(std::move(combine_fn.extractOutput(&accumulator_list[idx])), wnid_list[idx]);
              // printf("stage %d transform %d(Combine) task %d> push the resulf of wnid %d\n", tl_stage_id, tl_transform_id, tl_task_id, out_element.id);
              Serdes<WN<OutT>>::stream_serialize(out, out_element, this->is_eager);
            }
            wnid_list[idx] = wnid;
            combine_fn.resetAccumulator(&accumulator_list[idx]);
            combine_fn.addInput(&accumulator_list[idx], in_element.e);
          }
        }
      } catch (ChannelClosedException e) {
        assert(aggregated_in.eos());
        for(int idx=0; idx<num_windows; idx++) {
          WN<OutT> out_element(std::move(combine_fn.extractOutput(&accumulator_list[idx])), wnid_list[idx]);
          Serdes<WN<OutT>>::stream_serialize(out, out_element, this->is_eager);
          wnid_list[idx] = -1;
          combine_fn.resetAccumulator(&accumulator_list[idx]);
        }
      }
    }
  };

  template <typename CombineFnType>
  static PCollection<WN<typename CombineFnType::OutputType>>* globally(PCollection<WN<typename CombineFnType::InputType>>* pcollection, const CombineFnType& combine_fn) {
    return pcollection->apply(new WindowedCombineTransform<CombineFnType,typename CombineFnType::InputType,typename CombineFnType::OutputType>(combine_fn));
  }
};

