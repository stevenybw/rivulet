#ifndef RIVULET_CHANNEL_H
#define RIVULET_CHANNEL_H

#include "util.h"

/*
struct InputChannel
{
  virtual void* pull(size_t bytes)=0;
  virtual bool eos()=0;
};

struct OutputChannel
{
  virtual void push(void* buf, size_t bytes)=0;
  virtual void close()=0;
};

namespace LocalChannel
{
  const size_t channel_bytes = 8192;
  const size_t ALIGNMENT     = 64
  struct SenderPrivate {
    int    seq;
    size_t bytes;
  };

  struct BufferWindow {
    char data[2][channel_bytes];
  };

  struct Request {
    int       seq;
    size_t    bytes;
    uintptr_t ptr;
  };

  struct alignas(ALIGNMENT) Sync {
    alignas(64) volatile Request req; // request
    alignas(64) volatile int     seq; // response
  };

  struct LocalInputChannel : public InputChannel
  {
    size_t offset;
    size_t buf_size;
    char*  buf;
    size_t staging_area_capacity;
    char*  staging_area;
    std::shared_ptr<BufferWindow> _window;
    std::shared_ptr<Sync>         _sync;

    LocalInputChannel(std::shared_ptr<BufferWindow> window, std::shared_ptr<Sync> sync) :
        offset(0),
        bytes(0),
        buf(NULL),
        _window(window),
        _sync(sync) {
      staging_area_capacity = channel_bytes;
      staging_area = new char[channel_bytes];    
    }

    void* pull(size_t bytes) override {
      if (offset + bytes <= buf_size) {
        void* result = &buf[offset];
        offset += bytes;
        return result;
      } else {
        size_t received_bytes = 0;
        size_t curr_bytes = buf_size - offset;
        memcpy(&staging_area[received_bytes], &buf[offset], curr_bytes);
        while (_sync->seq == _sync->req.seq) {
          // yield(); TODO
        }
        while (true) {
          received_bytes += curr_bytes;
          offset   = 0;
          buf_size = _sync->req.bytes;
          buf = (char*) _sync->req.ptr;
        }
      }
    }
  };

  struct LocalOutputChannel : public OutputChannel
  {
    std::unique_ptr<SenderPrivate> _sender_private;
    std::shared_ptr<BufferWindow>  _window;
    std::shared_ptr<Sync>          _sync;
  };
};

struct ChannelMgr 
{
  static InputChannel* create_input_channel(int from_rank, int from_lid) {
    return NULL; //TODO
  }
  static OutputChannel* create_output_channel(int to_rank, int to_lid) {
    return NULL; //TODO
  }
  static std::tuple<LocalOutputChannel*, LocalInputChannel*> create_local_channels() {
    LocalOutputChannel* out = NULL;
    LocalInputChannel* in = NULL;
    return std::make_tuple(out, in); //TODO
  }
};
*/

template<size_t channel_bytes, bool use_nts = false, size_t ALIGNMENT=64>
struct Channel_1 {
  struct SenderPrivate {
    int    seq;
    size_t bytes;
  };

  struct BufferWindow {
    char data[2][channel_bytes];
  };

  struct Request {
    int       seq;
    size_t    bytes;
    uintptr_t ptr;
  };

  struct alignas(ALIGNMENT) Sync {
    alignas(64) volatile Request req; // request
    alignas(64) volatile int     seq; // response
  };

  SenderPrivate* _sender_private;
  
  // sender owned buffers
  BufferWindow* _window;

  // inter-thread communication
  Sync* _sync;

  // closed flag;
  bool* _closed;

  Channel_1() {
    _sender_private = NULL;
    _window = NULL;
    _sync = NULL;
    _closed = NULL;
  }

  void init() {
    _window = (BufferWindow*) am_memalign(channel_bytes, sizeof(BufferWindow));
    _sender_private = (SenderPrivate*) am_memalign(ALIGNMENT, sizeof(SenderPrivate));
    _sender_private->seq   = 0;
    _sender_private->bytes = 0;
    _sync   = (Sync*) am_memalign(ALIGNMENT, sizeof(Sync));
    _sync->req.seq = 0;
    _sync->seq     = 0;
    _closed = (bool*) malloc(sizeof(bool));
    *_closed = false;
    assert((uintptr_t)(&(_window->data[0])) % ALIGNMENT == 0);
    assert((uintptr_t)(&(_window->data[1])) % ALIGNMENT == 0);
  }


  /*
  void init(SenderPrivate* sender_private) {
    _window = (BufferWindow*) am_memalign(channel_bytes, sizeof(BufferWindow));
    _sender_private = sender_private;
    _sender_private->seq   = 0;
    _sender_private->bytes = 0;
    _sync   = (Sync*) am_memalign(ALIGNMENT, sizeof(Sync));
    _sync->req.seq = 0;
    _sync->seq     = 0;
    _closed = malloc(sizeof(bool));
    *_closed = false;
    assert((uintptr_t)(&(_window->data[0])) % ALIGNMENT == 0);
    assert((uintptr_t)(&(_window->data[1])) % ALIGNMENT == 0);
  }

  void init(BufferWindow* _window, SenderPrivate* sender_private) {
    _window = _window;
    _sender_private = sender_private;
    _sender_private->seq   = 0;
    _sender_private->bytes = 0;
    _sync   = (Sync*) am_memalign(ALIGNMENT, sizeof(Sync));
    _sync->req.seq = 0;
    _sync->seq     = 0;
    _closed = malloc(sizeof(bool));
    *_closed = false;
    assert((uintptr_t)(&(_window->data[0])) % ALIGNMENT == 0);
    assert((uintptr_t)(&(_window->data[1])) % ALIGNMENT == 0);
  }
  */

  template <typename T>
  size_t push_explicit(const T& data, size_t curr_bytes) {
    assert(!*_closed);
    size_t bytes = sizeof(T);
    int seq = _sender_private->seq;
    if (curr_bytes + bytes > channel_bytes) {
      _sender_private->seq = seq^1;
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    }
    if (use_nts) {
      memcpy_nts(&(_window->data[seq][curr_bytes]), &data, bytes);
    } else {
      memcpy(&(_window->data[seq][curr_bytes]), &data, bytes);
    }
    curr_bytes += bytes;

    return curr_bytes;
  }

  template <typename T>
  size_t push_explicit(const char* data, size_t bytes, size_t curr_bytes) {
    assert(!*_closed);
    int seq = _sender_private->seq;
    if (curr_bytes + bytes > channel_bytes) {
      _sender_private->seq = seq^1;
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    }
    if (use_nts) {
      memcpy_nts(&(_window->data[seq][curr_bytes]), data, bytes);
    } else {
      memcpy(&(_window->data[seq][curr_bytes]), data, bytes);
    }
    curr_bytes += bytes;

    return curr_bytes;
  }

  size_t flush_and_wait_explicit(size_t curr_bytes) {
    assert(!*_closed);
    int seq = _sender_private->seq;
    if (curr_bytes > 0) {
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      // _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    }
    wait_for([this, seq](){return _sync->seq == seq;});
    _sender_private->seq = seq;
    return 0;
  }

  template <typename T>
  void push(const T& data) {
    assert(!*_closed);
    size_t bytes = sizeof(T);
    int seq = _sender_private->seq;
    size_t curr_bytes = _sender_private->bytes;
    if (curr_bytes + bytes > channel_bytes) {
      _sender_private->seq = seq^1;
      _sender_private->bytes = bytes;
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    } else {
      _sender_private->bytes = curr_bytes + bytes;
    }
    if (use_nts) {
      memcpy_nts(&(_window->data[seq][curr_bytes]), &data, bytes);
    } else {
      memcpy(&(_window->data[seq][curr_bytes]), &data, bytes);
    }
  }

  void push(const char* data, size_t bytes) {
    assert(!*_closed);
    int seq = _sender_private->seq;
    size_t curr_bytes = _sender_private->bytes;
    if (curr_bytes + bytes > channel_bytes) {
      _sender_private->seq = seq^1;
      _sender_private->bytes = bytes;
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    } else {
      _sender_private->bytes = curr_bytes + bytes;
    }
    // _mm_stream_ps((float*) &(_window->data[seq][curr_bytes]), *((__m128*) &data));
    memcpy(&(_window->data[seq][curr_bytes]), data, bytes);
  }

  void flush_and_wait() {
    assert(!*_closed);
    int seq = _sender_private->seq;
    size_t curr_bytes = _sender_private->bytes;
    if (curr_bytes > 0) {
      wait_for([this, seq](){return _sync->seq == seq;});
      _mm_sfence();
      _sync->req.bytes = curr_bytes;
      _sync->req.ptr   = (uintptr_t) &(_window->data[seq]);
      _sync->req.seq   = seq^1;
      // _mm_sfence();
      seq ^= 1;
      curr_bytes = 0;
    }
    wait_for([this, seq](){return _sync->seq == seq;});
    _sender_private->seq = seq;
    _sender_private->bytes = curr_bytes;
  }

  void close() {
    flush_and_wait();
    *_closed = true;
  }

  bool eos() {
    return *_closed;
  }

  template <typename Callable>
  //void poll(const std::function<void(const char* data, size_t bytes)>& process_callback) {
  void poll (const Callable& process_callback) {
    if (_sync->seq != _sync->req.seq) {
      process_callback((const char*) _sync->req.ptr, _sync->req.bytes);
      _sync->seq = _sync->req.seq;
    }
  }
};

#endif