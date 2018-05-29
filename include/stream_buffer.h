/*
 * Stream Buffer Implementation
 *
 * Copyright 2018 Bowen Yu, Tsinghua University
 * 
 */

#ifndef STREAM_BUFFER_H
#define STREAM_BUFFER_H

#include <utility>

#include <pthread.h>
#include <malloc.h>

#include "util.h"

class StreamBufferHandler {
public:
  virtual void on_issue(int buffer_id, char* buffer, size_t bytes)=0;
  virtual void on_wait(int buffer_id)=0;
};

/*! \brief <DEPRECATED> StreamBuffer<T> provides an abstraction for multiple threads streaming write and process.
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
template <typename HandlerType,
          size_t num_buffers = 2,
          size_t minibatch_buffer_size_per_partition=4096>
class StreamBuffer
{
  /*! \brief Index
   *
   *  Shared structures used for synchronization
   */
  struct Index 
  {
    volatile uint64_t commit_pos;
    volatile uint64_t request_pos;

    void reset() {
      // order matters here: should reset commit before allow other threads to retry
      commit_pos = 0;
      request_pos = 0;
    }
    volatile uint64_t query_request() { return request_pos; }
    volatile uint64_t query_commit() { return commit_pos; }

    // Request bytes, return current position
    volatile uint64_t faa_request(uint64_t bytes) {
      return __sync_fetch_and_add(&request_pos, bytes);
    }

    // Commit bytes, return current position
    volatile uint64_t faa_commit(uint64_t bytes) {
      return __sync_fetch_and_add(&commit_pos, bytes);
    }
  };

private:
  size_t      capacity;
  HandlerType handler;
  char*       buffer_pool[num_buffers];
  Index*      index;
  volatile uint64_t buffer_pool_consumed;
  volatile uint64_t buffer_pool_produced;
  char*    volatile active_buffer;

public:
  struct ThreadContext {
    int    minibatch_bytes;
    char*  minibatch;
    ThreadContext() {
      minibatch_bytes = 0;
      minibatch = (char*) am_memalign(4096, minibatch_buffer_size_per_partition);
    }
    ~ThreadContext() {
      free(minibatch); 
    }

    bool empty() { return get_bytes() == 0; }
    char* get_data() { return minibatch; }
    size_t get_bytes() { return minibatch_bytes; }

    /*! \brief Append an memory region into minibatch (assume checked)
     */
    void append_checked(const void* data, size_t el_bytes) { 
      memcpy(&minibatch[get_bytes()], data, el_bytes);
      minibatch_bytes += el_bytes;
    }

    /*! \brief Clear the cached minibatch
     */
    void clear() {
      minibatch_bytes = 0;
    }
  };

  StreamBuffer() : capacity(0), buffer_pool {nullptr}, buffer_pool_consumed(0), buffer_pool_produced(0), active_buffer(nullptr), index(nullptr) {}
  StreamBuffer(StreamBuffer&& buffer) {
    swap(buffer);
  }

  StreamBuffer& operator=(StreamBuffer&& buffer) {
    swap(buffer);
    return *this;
  }

  void swap(StreamBuffer& buffer) {
    std::swap(capacity, buffer.capacity);
    std::swap(handler, buffer.handler);
    std::swap(buffer_pool, buffer.buffer_pool);
    std::swap(buffer_pool_consumed, buffer.buffer_pool_consumed);
    std::swap(buffer_pool_produced, buffer.buffer_pool_produced);
    std::swap(active_buffer, buffer.active_buffer);
    std::swap(index, buffer.index);
  }

  /*! \brief Initialize the stream buffer with specified capacity in bytes
   */
  StreamBuffer(size_t capacity, HandlerType&& handler) : capacity(capacity), handler(std::move(handler)) {
    for(uint64_t i=0; i<num_buffers; i++) {
      buffer_pool[i] = (char*) memory::allocate_shared_rw(capacity);
    }
    buffer_pool_consumed = 0;
    buffer_pool_produced = 0;
    active_buffer = buffer_pool[0];
    index = (Index*) memory::allocate_shared_rw(sizeof(Index));
    index->reset();
  }

  ~StreamBuffer() {
    for(uint64_t i=0; i<num_buffers; i++) {
      if (buffer_pool[i] != nullptr) {
        memory::free(buffer_pool[i]);
      }
    }
    if (index != nullptr) {
      memory::free(index);
    }
  }

  void on_issue(int buffer_id, char* buffer, size_t bytes) { handler.on_issue(buffer_id, buffer, bytes); }
  void on_wait(int buffer_id) { handler.on_wait(buffer_id); }

  /*! \brief Writeback ThreadContext
   *
   *  Write back the data in thread local buffer into the stream buffer, and
   *  clear the data in thread local buffer
   */
  int writeback_thread_context(ThreadContext& ctx) {
    size_t bytes = ctx.get_bytes();
    if (bytes == 0) {
      return 0;
    }
label_retry:
    uint64_t request_pos = index->faa_request(bytes);
    if (request_pos + bytes > capacity) {
      // if full, process & retry
      if (request_pos <= capacity) { // master
        // wait for completion before issue this buffer
        wait_for([this, request_pos](){return (index->query_commit() == request_pos); });
        int current_buffer_id = buffer_pool_produced % num_buffers;
        PRINTF("%d> writeback_thread_context on_issue  buffer_pool_produced=%ld\n", g_rank, buffer_pool_produced);
        on_issue(current_buffer_id, active_buffer, request_pos);
        // ensure produced+1 can be write as active buffer
        if (buffer_pool_produced + 1 - buffer_pool_consumed >= num_buffers) {
          int buffer_id = buffer_pool_consumed % num_buffers;
          on_wait(buffer_id);
          buffer_pool_consumed++;
        }
        buffer_pool_produced++;
        int buffer_id = buffer_pool_produced % num_buffers;
        active_buffer = buffer_pool[buffer_id];
        index->reset();
        PRINTF("%d> writeback_thread_context complete  buffer_pool_produced=%ld\n", g_rank, buffer_pool_produced);
      } else { // follower
        // master process done when index->query_request() <= capacity
        wait_for([this](){return (index->query_request() <= capacity);});
      }
      goto label_retry; // because it does not have effects, it is safe to retry
    } else {
      memcpy(&active_buffer[request_pos], ctx.get_data(), bytes); // use non-temporal store here
      index->faa_commit(bytes);
      ctx.clear();
    }
    return 0;
  }

  /*! \brief Push elem into the buffer
   */
  template <typename T>
  int push(ThreadContext& ctx, const T& elem) {
    if (ctx.get_bytes() + sizeof(elem) <= minibatch_buffer_size_per_partition) { // CANDIDATE 1
      ctx.append_checked(&elem, sizeof(elem));
    } else {
      writeback_thread_context(ctx);
      assert(ctx.empty());
      ctx.append_checked(&elem, sizeof(elem));
    }

    return 0;
  }

  /*! \brief [GroupOp] A pthread barrier is implied so that we are sure that all participating threads has flushed
   *
   */
  int flush(ThreadContext& ctx, pthread_barrier_t* barrier) {
    if (ctx.get_bytes() > 0) {
      writeback_thread_context(ctx);
      assert(ctx.empty());
    }
    // wait for everyone to write back, and the serial thread clear the buffer
    int ret = pthread_barrier_wait(barrier);
    if (ret == PTHREAD_BARRIER_SERIAL_THREAD) {
      assert(index->query_request() == index->query_commit());
      size_t request_pos = index->query_request();
      index->reset();
      PRINTF("%d> flush on_issue  buffer_pool_produced=%ld\n", g_rank, buffer_pool_produced);
      int current_buffer_id = buffer_pool_produced % num_buffers;
      on_issue(current_buffer_id, active_buffer, request_pos);
      buffer_pool_produced++;
      assert(buffer_pool_produced - buffer_pool_consumed <= num_buffers);
      while (buffer_pool_consumed < buffer_pool_produced) {
        on_wait(buffer_pool_consumed % num_buffers);
        buffer_pool_consumed++;
      }
      PRINTF("%d> flush complete  buffer_pool_produced=%ld\n", g_rank, buffer_pool_produced);
      buffer_pool_consumed = 0;
      buffer_pool_produced = 0;
      active_buffer = buffer_pool[0];
    } else {
      assert(ret == 0);
    }
    pthread_barrier_wait(barrier);
    // the buffer is in a clean state

    return 0;
  }
};

#endif