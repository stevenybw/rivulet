#g++ -std=c++14 -pthread -O0 -g -I../include -o test_fs_monitor test_fs_monitor.cc
#mpicxx -std=c++14 -pthread -O2 -g -I../include -o test_stream_buffer_mpi test_stream_buffer_mpi.cc ../src/util.cc -lnuma
mpicxx -march=native -std=c++14 -pthread -fopenmp -O2 -g -I../include -o test_new_stream_buffer test_new_stream_buffer.cc ../src/util.cc -lnuma
#g++ -std=c++14 -pthread -O2 -g -o dram_benchmark dram_benchmark.cc