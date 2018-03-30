struct __attribute__((packed)) Header {
  int    type;
  size_t bytes;
};

enum {
  TYPE_STRING = 1,
  TYPE_TS     = 2,
  TYPE_WN     = 3
};

template <typename T>
struct Serdes
{
  static T stream_deserialize(InputChannel* in) {
    cerr << "[Error] Not Implemented Serdes" << endl;
    assert(false);
  }

  static void stream_serialize(OutputChannel* out, T& elem) {
    cerr << "[Error] Not Implemented Serdes"
    assert(false);
  }
};

// Match TS type via partial specification
template <typename T>
struct Serdes<TS<T>>
{
  static TS<T> stream_deserialize(InputChannel* in) {
    Header* buf = (Header*) in->pull(sizeof(Header));
    assert(buf->type == TYPE_TS);
    size_t ts = buf->bytes;
    T elem = Serdes<T>::stream_deserialize(in);
    return TS<T>(std::move(elem), ts);
  }

  static void stream_serialize(OutputChannel* out, TS<T>& elem) {
    Header header = {TYPE_TS, elem.ts};
    out->push(&header, sizeof(header));
    Serdes<T>::stream_serialize(out, elem.e);
  }
};

template <typename T>
struct Serdes<WN<T>>
{
  static WN<T> stream_deserialize(InputChannel* in) {
    Header* buf = (Header*) in->pull(sizeof(Header));
    assert(buf->type == TYPE_WN);
    size_t id = buf->bytes;
    T elem = Serdes<T>::stream_deserialize(in);
    return WN<T>(std::move(elem), id);
  }

  static void stream_serialize(OutputChannel* out, WN<T>& elem) {
    Header header = {TYPE_WN, elem.id};
    out->push(&header, sizeof(header));
    Serdes<T>::stream_serialize(out, elem.e);
  }
};

template <>
struct Serdes<string>
{
  static string stream_deserialize(InputChannel* in) {
    Header* buf = (Header*) in->pull(sizeof(Header));
    assert(buf->type == TYPE_STRING);
    size_t bytes = buf->bytes;
    char* str = (char*) in->pull(bytes);
    return string(str, bytes);
  }

  static void stream_serialize(OutputChannel* out, string& elem) {
    Header header = {TYPE_STRING, elem.size()};
    out->push(&header, sizeof(header));
    out->push(elem.c_str(), elem.size());
  }
};

template <>
struct Serdes<double>
{
  static double stream_deserialize(InputChannel* in) {
    double* ptr = (double*) in->pull(sizeof(double));
    return *ptr;
  }

  static void stream_serialize(OutputChannel* out, double& elem) {
    out->push(&elem, sizeof(double));
  }
};

template <typename InputT, typename OutputT>
struct DoFn {
  typedef InputT InputType;
  typedef OutputT OutputType;

  struct ProcessContext {
    InputT&        _element;
    OutputChannel* _output_channel;
    ProcessContext(InputT& element, OutputChannel* output_channel) : _element(element), _output_channel(output_channel) {}
    
    InputT& element() { return _element; }

    void output(OutputT&& output_element) {
      Serdes<OutputT>::stream_serialize(_output_channel, output_element);
    }
  };
  virtual void processElement(ProcessContext& processContext)=0;
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
          typename DoFnType::ProcessContext pc(in_element, out);
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
          // printf("!!  %d send to %d\n", ((PTInstance*)state)->instance_id, dest);
          Serdes<InputT>::stream_serialize(outputs[dest], in_element);
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
        Serdes<string>::stream_serialize(out, line);
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
            Serdes<WN<T>>::stream_serialize(out, out_element);
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

// Here we assume strict ordering, thus for each source, the window id mono-increase
// Aggregate each window and emit the result into subsequent transforms. This is 
// N->1  transform
struct WindowedCombine
{
  template <typename CombineFnType, typename InT, typename OutT>
  struct WindowedCombineTransform : public TaggedPTransform<WN<InT>, WN<OutT>>
  {
    CombineFnType combine_fn;
    WindowedCombineTransform(const CombineFnType& combine_fn) : combine_fn(combine_fn) {}

    string type_name() override { return "WindowedCombineTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      PTInstance* pti = new PTInstance;
      pti->instance_id = instance_id;
      pti->ptransform  = this;
      pti->state       = NULL;
      return pti;
    }

    void execute (const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(outputs.size() == 1);
      OutputChannel* out = outputs[0];
      int num_inputs = inputs.size();
      bool* input_valid_list = new bool[num_inputs];
      WN<InT>* staged_inputs = new WN<InT>[num_inputs];
      int closed_inputs = 0;
      uint64_t curr_wnid = (uint64_t) -1;
      for(int i=0; i<num_inputs; i++) {
        try {
          staged_inputs[i] = std::move(Serdes<WN<InT>>::stream_deserialize(inputs[i]));
          input_valid_list[i] = true;
          curr_wnid = min(curr_wnid, staged_inputs[i].id);
        } catch (ChannelClosedException& e) {
          assert(inputs[i]->eos());
          input_valid_list[i] = false;
          closed_inputs++;
        }
      }
      if (closed_inputs < num_inputs) {
        typename CombineFnType::AccumulatorType* curr_acc = combine_fn.createAccumulator();
        while (closed_inputs < num_inputs) {
          uint64_t min_wnid = (uint64_t)-1;
          for (int i=0; i<num_inputs; i++) {
            if (input_valid_list[i] && (staged_inputs[i].id == curr_wnid)) {
              // printf("received from %d, wnid = %llu, val = %lf\n", i, staged_inputs[i].id, staged_inputs[i].e);
              // WN<InT> in_element = std::move(staged_inputs[i]);
              combine_fn.addInput(curr_acc, staged_inputs[i].e);
              min_wnid = min(min_wnid, staged_inputs[i].id);
              try {
                staged_inputs[i] = Serdes<WN<InT>>::stream_deserialize(inputs[i]);
              } catch (ChannelClosedException& e) {
                // printf("CLOSED %d\n", i);
                assert(inputs[i]->eos());
                input_valid_list[i] = false;
                closed_inputs++;
              }
            } else if (input_valid_list[i]) {
              min_wnid = min(min_wnid, staged_inputs[i].id);
            }
          }
          assert(min_wnid >= curr_wnid);
          if (min_wnid > curr_wnid) {
            WN<OutT> out_element(std::move(combine_fn.extractOutput(curr_acc)), curr_wnid);
            Serdes<WN<OutT>>::stream_serialize(out, out_element);
            curr_wnid = min_wnid;
            combine_fn.resetAccumulator(curr_acc);
          }
        }
        assert(closed_inputs == num_inputs);
        WN<OutT> out_element(std::move(combine_fn.extractOutput(curr_acc)), curr_wnid);
        Serdes<WN<OutT>>::stream_serialize(out, out_element);
        combine_fn.resetAccumulator(curr_acc);
      }
      out->close();
    }
  };

  template <typename CombineFnType>
  static PCollection<WN<typename CombineFnType::OutputType>>* globally(PCollection<WN<typename CombineFnType::InputType>>* pcollection, const CombineFnType& combine_fn) {
    return pcollection->apply(new WindowedCombineTransform<CombineFnType,typename CombineFnType::InputType,typename CombineFnType::OutputType>(combine_fn));
  }
};

