/*
 * Stream Buffer Implementation
 *
 * Copyright 2018 Bowen Yu, Tsinghua University
 * 
 */

/*! \brief StreamBuffer<T> provides an abstraction for multiple threads streaming write and process.
 *  Multiple producers may call update
 *  Template
 *    minibatch_buffer_size_per_partition: thread-local minibatch for the stream buffer
 *
 *  Callbacks:
 *    on_issue(int buffer_id, char* buffer, size_t bytes);
 *    on_wait(int buffer_id);
 *  
 *  push(ctx, elem)   push an element to the streaming buffer
 *  flush(ctx)
 */
template <typename T,
          typename HandlerType,
          size_t num_buffers = 2;
          size_t minibatch_buffer_size_per_partition=4096>
class StreamBuffer
{
private:
  size_t      capacity;
  HandlerType handler;
  char*       buffer_pool[num_buffers];
  uint64_t    buffer_pool_consumed;
  uint64_t    buffer_pool_produced;
  char*       active_buffer;
  size_t*     size;

public:
  struct ThreadContext {
    int   minibatch_bytes; 
    char  minibatch[minibatch_buffer_size_per_partition];
    ThreadContext() : minibatch_bytes(0) {}

    bool empty() { return minibatch_bytes == 0; }
    char* get_data() { return minibatch; }
    size_t get_bytes() { return minibatch_bytes; }

    /*! \brief Append an memory region into minibatch (assume checked)
     */
    void append_checked(const void* data, size_t el_bytes) { 
      memcpy(&minibatch[minibatch_bytes], data, el_bytes);
      minibatch_bytes += el_bytes;
    }
  };

  /*! \brief Initialize the stream buffer with specified capacity in bytes
   */
  StreamBuffer(size_t capacity, HandlerType&& handler) : capacity(capacity), handler(handler) {
    for(uint64_t i=0; i<num_buffers; i++) {
      buffer_pool[i] = (char*) memory::allocate_shared_rw(capacity);
    }
    buffer_pool_consumed = 0;
    buffer_pool_produced = 0;
    active_buffer = buffer_pool[0];
    size = (size_t*) memory::allocate_shared_rw(sizeof(size_t));
    (*size) = 0;
  }

  ~StreamBuffer() {
    for(uint64_t i=0; i<num_buffers; i++) {
      memory::free(buffer_pool[i]);
    }
    memory::free(size);
  }

  void on_issue(int buffer_id, char* buffer, size_t bytes) { handler.on_issue(buffer_id, buffer, bytes); }
  void on_wait(int buffer_id) { handler.on_wait(buffer_id); }

  /*! \brief Writeback ThreadContext
   *
   *  Write back the data in thread local buffer into the stream buffer, and
   *  clear the data in thread local buffer
   */
  int writeback_thread_context(ThreadContext& ctx) {
label_retry:
    size_t bytes = ctx.get_bytes();
    size_t curr_size = __sync_fetch_and_add(size, bytes);
    if (curr_size + bytes > capacity) {
      // if full, process & retry
      if (curr_size <= capacity) { // master
        int current_buffer_id = buffer_pool_produced % num_buffers;
        on_issue(current_buffer_id, active_buffer, curr_size);
        // ensure produced+1 can be write as active buffer
        if (buffer_pool_produced + 1 - buffer_pool_consumed >= num_buffers) {
          int buffer_id = buffer_pool_consumed % num_buffers;
          on_wait(buffer_id);
          buffer_pool_consumed++;
        }
        buffer_pool_produced++;
        int buffer_id = buffer_pool_produced % num_buffers;
        active_buffer = buffer_pool[buffer_id];
        (*size) = 0; // enable access
      } else { // follower
        // master process done when (*size) <= capacity
        wait_for([size](){return ((*size) <= capacity);});
      }
      goto label_retry; // because it does not have effects, it is safe to retry
    } else {
      memcpy_nts(&buffer[curr_size], ctx.get_data(), bytes); // use non-temporal store here
      ctx.clear();
    }
    return 0;
  }

  /*! \brief Push elem into the buffer
   */
  int push(ThreadContext& ctx, const T& elem) {
    if (ctx.get_bytes() + sizeof(elem) <= minibatch_buffer_size_per_partition) {
      ctx.append_checked(&elem, sizeof(elem));
    } else {
      writeback_thread_context(ctx);
      assert(ctx.empty());
      ctx.append_checked(&elem, sizeof(elem));
    }
    return 0;
  }

  /*! \brief [GroupOp] A barrier is implied so that we are sure that all participating threads has flushed
   *
   */
  int flush(ThreadContext& ctx, pthread_barrier_t* barrier) {
    if (ctx.get_bytes() > 0) {
      writeback_thread_context(ctx);
      assert(ctx.empty());
    }
    int ret = pthread_barrier_wait(barrier);
    if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
      size_t curr_size = (*size);
      (*size) = 0;
      int current_buffer_id = buffer_pool_produced % num_buffers;
      on_issue(current_buffer_id, active_buffer, curr_size);
      buffer_pool_produced++;
      assert(buffer_pool_produced - buffer_pool_consumed <= num_buffers);
      while (buffer_pool_consumed < buffer_pool_produced) {
        on_wait(buffer_pool_consumed % num_buffers);
        buffer_pool_consumed++;
      }
    } else {
      assert(ret == 0);
    }
    buffer_pool_consumed = 0;
    buffer_pool_produced = 0;
    active_buffer = buffer_pool[0];
  }
};