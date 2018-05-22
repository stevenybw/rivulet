#ifndef RIVULET_FILE_H
#define RIVULET_FILE_H

#define _GNU_SOURCE

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
  int    _mmap_prot;
  int    _mmap_flags;
  size_t _bytes;
  void*  _addr;

  size_t get_bytes() { return _bytes; }
  void* get_addr() { return _addr; }

  bool create(const char* path, size_t bytes) {
    if (access(path, F_OK) != -1) {
      printf("WARNING: Path to be created %s exists, which would be overwritten\n", path);
      ::unlink(path);
    }
    mode_t mode = S_IRUSR | S_IWUSR | S_IRGRP | S_IROTH;
    int fd = ::open(path, O_RDWR | O_CREAT, mode);
    if (fd < 0) {
      cout << "cannot open file " << path;
      perror("file open failed");
      assert(false);
    }
    if (ftruncate(fd, bytes) < 0) {
      perror("ftruncate");
      assert(false);
    }
    ::close(fd);
    return true;
  }

  void* _mmap(int access_pattern, size_t bytes, int mmap_prot, int mmap_flags, int fd) {
    if (bytes == 0) {
      return NULL;
    } else {
      void* addr = mmap(NULL, bytes, mmap_prot, mmap_flags, fd, 0);
      if (addr == MAP_FAILED) {
        perror("mmap");
        assert(false);
      }
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
      return addr;
    }
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
      cout << "cannot open file " << path << endl;
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
    
    void* addr = _mmap(access_pattern, bytes, mmap_prot, mmap_flags, fd);

    _opened    = true;
    _path      = path;
    _fd        = fd;
    _file_mode = mode;
    _access_pattern = access_pattern;
    _mmap_prot = mmap_prot;
    _mmap_flags= mmap_flags;
    _bytes     = bytes;
    _addr      = addr;
    return true;
  }

  void msync() {
    int ok;
    ok = ::msync(_addr, _bytes, MS_SYNC);
    if (ok < 0) {
      perror("msync");
      assert(false);
    }
    ok = ::fsync(_fd);
    if (ok < 0) {
      perror("fsync");
      assert(false);
    }
  }

  void unlink() {
    printf("Unlink file %s\n", _path.c_str());
    int ok = ::unlink(_path.c_str());
    if (ok < 0) {
      perror("unlink");
      assert(false);
    }
  }

  void* resize(size_t new_size) {
    int ok;
    ok = ftruncate(_fd, new_size);
    if (ok < 0) {
      perror("ftruncate");
      assert(false);
    }
    void* addr = NULL;
    if (_addr == NULL) {
      addr = _mmap(_access_pattern, new_size, _mmap_prot, _mmap_flags, _fd);
    } else {
      addr = mremap(_addr, _bytes, new_size, MREMAP_MAYMOVE);
      if (addr == MAP_FAILED) {
        perror("mremap");
        assert(false);
      }
    }
    _addr = addr;
    _bytes = new_size;

    return _addr;
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