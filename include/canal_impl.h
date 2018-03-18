template <typename InputT, typename OutputT>
struct DoFn {
  typedef InputT InputType;
  typedef OutputT OutputType;

  struct ProcessContext {
    InputT&        _element;
    OutputChannel* _output_channel;
    ProcessContext(InputT& element, OutputChannel* output_channel) : _element(element), _output_channel(output_channel) {}
    
    InputT& element() { return _element; }

    void output(OutputT& output_element) {
      Sedes::stream_serialize(out, output_element);
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
      while (!in->eos()) {
        InputT in_element = Sedes::stream_deserialize<InputT>(in);
        ProcessContext pc(in_element, out);
        do_fn.processElement(pc);
      }
    }
  }

  template <typename DoFnType>
  static ParDoTransform<DoFnType>* of(const DoFnType& do_fn) {
    return new ParDoTransform(do_fn);
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
      instance->state = NULL;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      assert(inputs.size() == 1);
      InputChannel* in = inputs[0];
      int num_outputs = outputs.size();
      while (!in->eos()) {
        InputT in_element = Serdes::stream_deserialize<InputT>(in);
        int dest = (in_element.id) % num_outputs;
        Serdes::stream_serialize(outputs[dest], in_element);
      }
    }
  };
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

    string type_name() override { return "ReadTransform"; }
    bool is_elementwise() override { return true; }
    bool is_shuffle() override { return false; }

    PTInstance* create_instance(int instance_id) override {
      State* state = new State(path + "." + std::to_string(instance_id));
      if (!(state->fin)) {
        cerr << "failed to open file " << path << endl;
        assert(false);
      }
      PTInstance* instance = new PTInstance;
      instance->instance_id = instance_id;
      instance->ptransform = this;
      instance->state = (void*) state;
      return instance;
    }

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      InputChannel* in = inputs[0];
      OutputChannel* out = outputs[0];
      State* st = (State*) state;
      string line;
      std::ifstream& fin = st->fin;
      while (std::getline(fin, line)) {
        while(!out->push(line.c_str(), line.size())) {
          yield();
        }
      }
      assert(!fin);
      out->close();
    }

    /*
    bool progress(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      InputChannel* in  = inputs[0];
      OutputChannel* out = outputs[0];
      State*   st  = (State*) state;
      string line;
      std::ifstream& fin = st->fin;
      if (st->last_line_valid) {
        string& last_line = st->last_line;
        bool ok = out->push(last_line.c_str(), last_line.size());
        if (ok) {
          st->last_line_valid = false;
        } else {
          return true;
        }
      }
      while (std::getline(fin, line)) {
        bool ok = out->push(line.c_str(), line.size());
        if (!ok) {
          st->last_line_valid = true;
          st->last_line = std::move(line);
          break;
        }
      }
      if (!fin) {
        // EOF reached
        out->close();
        return false; // report completion
      }
    }*/
  };

  static ReadTransform* read() {
    return new ReadTransform;
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
        while (!in->eos()) {
          KV<T> in_element = Sedes::stream_deserialize<KV<T>>(in);
          uint64_t ts = in_element.ts;
          WN<T> out_element(std::move(in_element.e, ts / window_size));
          Sedes::stream_serialize(out, out_element);
        }
      }
    };

    template <typename T>
    static PCollection<WN<T>>* assign(PCollection<TS<T>>* pcollection, uint64_t window_size) {
      return pcollection->apply(FixedWindowsTransform<T>(window_size));
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

// combine each window, and output exactly one entry for each window
// there would be multiple sources from shuffle
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

    void execute(const InputChannelList& inputs, const OutputChannelList& outputs, void* state) override {
      OutputChannel* out = outputs[0];
      int num_inputs = inputs.size();
      bool* stage_valid = new bool[num_inputs];
      WN<InT>* staged_inputs = new WN<InT>[num_inputs];
      for(int i=0; i<num_inputs; i++) {
        if (!(inputs[i]->eos())) {
          stage_valid[i] = true;
          staged_inputs[i] = std::move(Serdes::stream_deserialize<WN<InT>>(inputs[i]));
        } else {
          stage_valid[i] = false;
        }
      }
      uint64_t curr_wnid = 0;
      typename CombineFnType::AccumulatorType* curr_acc = combine_fn.createAccumulator();
      while (true) {
        int closed_input = 0;
        uint64_t min_wnid = (uint64_t)-1;
        for(int i=0; i<num_inputs; i++) {
          if (stage_valid[i] && (staged_inputs[i].id == curr_wnid)) {
            // WN<InT> in_element = std::move(staged_inputs[i]);
            combine_fn.addInput(curr_acc, staged_inputs[i].e);
            if (inputs[i]->eos()) {
              stage_valid[i] = false;
            } else {
              staged_inputs[i] = Serdes::stream_deserialize<WN<InT>>(inputs[i]);
            }
            min_wnid = min(min_wnid, staged_inputs[i].id);
          } else if (stage_valid[i]) {
            min_wnid = min(min_wnid, staged_inputs[i].id);
          } else {
            closed_input++;
          }
        }
        assert(min_wnid >= curr_wnid);
        if (min_wnid > curr_wnid) {
          WN<OutT> out_element(std::move(combine_fn.extractOutput(curr_acc)), curr_wnid);
          Serdes::stream_serialize(out, out_element);
          curr_wnid = min_wnid;
          combine_fn.resetAccumulator(curr_acc);
        }
        if (closed_input == num_inputs) {
          WN<OutT> out_element(std::move(combine_fn.extractOutput(curr_acc)), curr_wnid);
          Serdes::stream_serialize(out, out_element);
          combine_fn.resetAccumulator(curr_acc);
          break;
        }
      }
    }
  };

  template <typename CombineFnType>
  static PCollection<WN<typename CombineFnType::OutputType>>* globally(PCollection<WN<typename CombineFnType::InputType>>* pcollection, const CombineFnType& combine_fn) {
    return pcollection->apply(WindowedCombineTransform<CombineFnType,typename CombineFnType::InputType,typename CombineFnType::OutputType>(combine_fn));
  }
};

