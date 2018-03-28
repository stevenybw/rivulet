#include <stdio.h>
#include <stdlib.h>
#include <immintrin.h>
#include <numa.h>
#include <numaif.h>

#include "util.h"

void* pages[128*1024];
int nodes[128*1024];
int statuses[128*1024];

void am_free(void* ptr) {
  free(ptr);
}

void* am_memalign(size_t align, size_t size) {
  void* ptr;
  int ok = posix_memalign(&ptr, align, size);
  assert(ok == 0);
  return ptr;
}

bool is_power_of_2(uint64_t num) {
  while (num>0 && (num&1)==0) {
    num >>= 1;
  }
  if (num == 1) {
    return true;
  } else {
    return false;
  }
}

void* malloc_pinned(size_t size) {
  // !NOTE: use memalign to allocate 256-bytes aligned buffer
  void* ptr = am_memalign(4096, size);
  if (mlock(ptr, size)<0) {
    perror("mlock failed");
    assert(false);
  }
  return ptr;
}

void memcpy_nts(void* dst, const void* src, size_t bytes) {
  if (bytes == 16) {
    _mm_stream_pd((double*)dst, *((const __m128d*)src));
  } else {
    __m256i* dst_p = (__m256i*) dst;
    const __m256i* src_p = (const __m256i*) src;
    size_t off = 0;
    // _mm_stream_ps((float*) &(_window->data[seq][curr_bytes]), *((__m128*) &data));
    for(; off<bytes; off+=32) {
      _mm256_stream_si256(dst_p, *src_p);
      dst_p++;
      src_p++;
    }
  }
}

void interleave_memory(void* ptr, size_t size, size_t chunk_size, int* node_list, int num_nodes) {
  printf("interleave begin");
  assert(chunk_size == 4096);
  char* buf = (char*) ptr;
  size_t count = 0;
  for(size_t pos=0; pos<size; pos+=chunk_size) {
    pages[count] = &buf[pos];
    nodes[count] = node_list[count % num_nodes];
    statuses[count] = -1;
    count++;
  }
  int ok = move_pages(0, count, pages, nodes, statuses, MPOL_MF_MOVE);
  assert(ok != -1);
  for(size_t i=0; i<count; i++) {
    assert(statuses[i] == node_list[i % num_nodes]);
  }
  printf("interleave end");
}

void pin_memory(void* ptr, size_t size) {
  if (mlock(ptr, size) < 0) {
    perror("mlock failed");
    assert(false);
  }
}