#include <stdlib.h>
#include <immintrin.h>
#include "util.h"

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

void* am_memalign(size_t align, size_t size) {
  void* ptr;
  int ok = posix_memalign(&ptr, align, size);
  assert(ok == 0);
  return ptr;
}

void am_free(void* ptr) {
  free(ptr);
}