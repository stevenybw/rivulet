#include <map>
#include <mutex>
#include <queue>
#include <string>
#include <vector>

#include <assert.h>

#include <errno.h>
#include <poll.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/inotify.h>
#include <unistd.h>

using namespace std;

class FileSystemWatch
{
  const char* LOCK_POSTFIX = ".completed.lock";
  std::mutex mu;
  int watch_fd;
  std::vector<string> paths;
  std::vector<int> wds;
  enum FileState { FILE_CREATED=1, FILE_OPENED=2 };
  std::map<string, FileState> file_list;
  std::queue<string> pending;

public:
  FileSystemWatch() : watch_fd(-1) {
    watch_fd = inotify_init();
    if (watch_fd == -1) {
      perror("inotify_init");
      exit(EXIT_FAILURE);
    }
  }

  ~FileSystemWatch() {
    close(watch_fd);    
  }

  void add_watch(string path) {
    std::lock_guard<std::mutex> lk(mu);
    paths.push_back(path);
    int wd = inotify_add_watch(watch_fd, path.c_str(), IN_OPEN | IN_CLOSE | IN_CREATE);
    if (wd == -1) {
      fprintf(stderr, "Cannot watch '%s'\n", path.c_str());
      perror("inotify_add_watch");
      exit(EXIT_FAILURE);
    }
    wds.push_back(wd);
  }

  bool lock_file(string path) {
    string lock_path = path + LOCK_POSTFIX;
    int fd = open(lock_path.c_str(), O_WRONLY | O_CREAT | O_EXCL, S_IRWXU|S_IRWXG);
    if (fd < 0) {
      return false;
    } else {
      close(fd);
      return true;
    }
  }

  string next_entry() {
    std::lock_guard<std::mutex> lk(mu);
    /* Some systems cannot read integer variables if they are not
      properly aligned. On other systems, incorrect alignment may
      decrease performance. Hence, the buffer used for reading from
      the inotify file descriptor should have the same alignment as
      struct inotify_event. */

    char buf[4096]
      __attribute__ ((aligned(__alignof__(struct inotify_event))));

    while(true) {
      if (pending.size() > 0) {
        string path = pending.front();
        pending.pop();
        if (!lock_file(path)) {
          continue;
        }
        return path;
      }
      ssize_t len = read(watch_fd, buf, sizeof buf);
      if (len == -1 && errno != EAGAIN) {
        perror("read");
        exit(EXIT_FAILURE);
      }
      if (len > 0) {
        const struct inotify_event *event = NULL;
        for (char* ptr = buf; ptr < buf + len; ptr += sizeof(struct inotify_event) + event->len) {
          event = (const struct inotify_event *) ptr;
          if (event->len > 0) {
            int wd = event->wd;
            string prefix;
            for (int i=0; i<wds.size(); i++) {
              if (wds[i] == wd) {
                prefix = paths[i];
              }
            }
            assert(prefix.size() > 0);
            string name = event->name;
            bool is_dir = event->mask & IN_ISDIR;
            if (is_dir) {
              // do not consider dir
              continue;
            }
            string path = prefix + name;
            if (event->mask & IN_CREATE) {
              // printf("CREATE %s\n", path.c_str());
              file_list[path] = FILE_CREATED;
            }
            if (event->mask & IN_OPEN) {
              // printf("OPEN %s\n", path.c_str());
              if (file_list.find(path) != file_list.end()) {
                // assert(file_list[path] == FILE_CREATED);
                file_list[path] = FILE_OPENED;
              }
            }
            if (event->mask & IN_CLOSE_WRITE) {
              // printf("CLOSE_WRITE %s\n", path.c_str());
              if (file_list.find(path) != file_list.end()) {
                assert(file_list[path] == FILE_OPENED);
                file_list.erase(path);
                if (path.find(LOCK_POSTFIX) == string::npos) {
                  pending.push(path);
                }
              }
            }
          }
        }
      }
    }
  }
};
