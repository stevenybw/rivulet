#ifndef RIVULET_FILE_H
#define RIVULET_FILE_H

#include <iostream>
#include <string>

#include <fcntl.h>
#include <stdio.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

using namespace std;

#define FILE_MODE_READ_ONLY 0
#define FILE_MODE_WRITE_ONLY 1
#define FILE_MODE_READ_WRITE 2

#define ACCESS_PATTERN_SEQUENTIAL 0
#define ACCESS_PATTERN_RANDOM 1
#define ACCESS_PATTERN_NORMAL 2

struct MappedFile {
  bool   _opened;
  string _path;
  int    _fd;
  int    _file_mode;
  int    _access_pattern;
  size_t _bytes;
  void*  _addr;

  bool create(const char* path, size_t bytes) {
    if (access(path, F_OK) != -1) {
      printf("ERROR: Path to be created %s exists\n", path);
      assert(false);
    }
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = ::open(path, O_RDWR | O_CREAT, mode);
    if (fd < 0) {
      cout << "cannot open file " << path;
      assert(false);
    }
    if (ftruncate(fd, bytes) < 0) {
      perror("ftruncate");
      assert(false);
    }
    ::close(fd);
    return true;
  }

  bool open(const char* path, int mode, int access_pattern, bool pre_load=false) {
    int open_flags;
    int mmap_flags;
    int mmap_prot;
    if (mode == FILE_MODE_READ_ONLY) {
      open_flags = O_RDONLY;
      mmap_prot  = PROT_READ;
      mmap_flags = MAP_PRIVATE;
      if (pre_load) {
        cout << "Populate" << endl;
        mmap_flags |= MAP_POPULATE;
      }
    } else if (mode == FILE_MODE_WRITE_ONLY) {
      open_flags = O_RDWR;
      mmap_prot  = PROT_READ | PROT_WRITE;
      mmap_flags = MAP_SHARED;
    } else if (mode == FILE_MODE_READ_WRITE) {
      open_flags = O_RDWR;
      mmap_prot  = PROT_READ | PROT_WRITE;
      mmap_flags = MAP_SHARED;
    } else {
      assert(false);
    }
    int fd = ::open(path, open_flags);
    if (fd < 0) {
      cout << "cannot open file " << path;
      assert(false);
    }
    size_t bytes = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    if (access_pattern == ACCESS_PATTERN_SEQUENTIAL) {
      if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL) != 0) {
        perror("posix_fadvise");
        assert(false);
      }
    } else if (access_pattern == ACCESS_PATTERN_RANDOM) {
      if (posix_fadvise(fd, 0, 0, POSIX_FADV_RANDOM) != 0) {
        perror("posix_fadvise");
        assert(false);
      }
    } else if (access_pattern == ACCESS_PATTERN_NORMAL) {
      // pass
    } else {
      assert(false);
    }
    void* addr = mmap(NULL, bytes, mmap_prot, mmap_flags, fd, 0);
    assert(addr != MAP_FAILED);
    if (access_pattern == ACCESS_PATTERN_SEQUENTIAL) {
      if (madvise(addr, bytes, MADV_SEQUENTIAL) < 0) {
        perror("madvise");
        assert(false);
      }
    } else if (access_pattern == ACCESS_PATTERN_RANDOM) {
      if (madvise(addr, bytes, MADV_RANDOM) < 0) {
        perror("madvise");
        assert(false);
      }
    } else if (access_pattern == ACCESS_PATTERN_NORMAL) {
      // pass
    } else {
      assert(false);
    }

    _opened    = true;
    _path      = path;
    _fd        = fd;
    _file_mode = mode;
    _access_pattern = access_pattern;
    _bytes     = bytes;
    _addr      = addr;
    return true;
  }

  void close() {
    if (_opened) {
      _opened = false;
      munmap(_addr, _bytes);
      _addr = NULL;
      ::close(_fd);
      _fd = -1;
      _path = "";
    }
  }
};

struct MappedMemory {
  bool   _valid;
  int    _access_pattern;
  size_t _bytes;
  void*  _addr;

  bool alloc(size_t bytes, int access_pattern) {
    int mmap_flags = MAP_PRIVATE | MAP_ANONYMOUS;
    //if (enable_huge_page) {
    //  mmap_flags |= MAP_HUGETLB;
    //}
    void* addr = mmap(NULL, bytes, PROT_READ|PROT_WRITE, mmap_flags, -1, 0);
    if (addr == MAP_FAILED) {
      perror("mmap");
      assert(false);
    }
    int advice = 0;
    if (access_pattern == ACCESS_PATTERN_SEQUENTIAL) {
      advice = MADV_SEQUENTIAL;
    } else if (access_pattern == ACCESS_PATTERN_RANDOM) {
      advice = MADV_RANDOM;
    }
    //if (enable_huge_page) {
    //  advice |= MADV_HUGEPAGE;
    //}
    if (advice!=0 && madvise(addr, bytes, advice)<0) {
      perror("madvise");
      assert(false);
    }

    _valid = true;
    _access_pattern = access_pattern;
    _bytes = bytes;
    _addr = addr;
    return true;
  }

  void free() {
    if (_valid) {
      _valid = false;
      munmap(_addr, _bytes);
      _bytes = 0;
      _addr  = NULL;
    }
  }
};

#endif