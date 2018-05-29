#include <thread>

using namespace std;

#include "fs_monitor.h"

void worker(FileSystemWatch& watch, int id) {
  while (true) {
    string path = watch.next_entry();
    printf("%d> new entry %s\n", id, path.c_str());
  }
}

int main(int argc, char* argv[])
{
  FileSystemWatch watch;
  for(int i=1; i<argc; i++) {
    string path = argv[i];
    printf("Subscribe %s\n", path.c_str());
    watch.add_watch(path);
  }

  thread t0(worker, std::ref(watch), 0);
  thread t1(worker, std::ref(watch), 1);
  t0.join();
  t1.join();

  return 0;
}