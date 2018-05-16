EXECUTABLES := page_rank_distributed bfs_distributed kmeans
#EXECUTABLES := stream_wordcount stream_sum page_rank graph_preprocess page_rank_distributed

HEADERS := include/rivulet.h include/rivulet_impl.h include/channel.h

all: $(EXECUTABLES)

clean:
	rm $(EXECUTABLES)

stream_wordcount: $(HEADERS) src/stream_wordcount.cc
	mpicxx -g -O2 -Iinclude -std=c++14 -march=native -fopenmp -o stream_wordcount src/util.cc src/stream_wordcount.cc -lnuma

stream_sum: $(HEADERS) src/stream_sum.cc
	mpicxx -g -O2 -Iinclude -std=c++14 -march=native -fopenmp -o stream_sum src/util.cc src/stream_sum.cc -lnuma

graph_preprocess: $(HEADERS) src/graph_preprocess.cc
	mpicxx -g -Wall -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o graph_preprocess src/util.cc src/ympi_alltoallv.c src/graph_preprocess.cc -lnuma

page_rank: $(HEADERS) src/page_rank.cc
	mpicxx -g -Wall -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o page_rank src/util.cc src/ympi_alltoallv.c src/page_rank.cc -lnuma

page_rank_distributed: $(HEADERS) src/page_rank_distributed.cc
	mpicxx -g -Wall -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o page_rank_distributed src/util.cc src/ympi_alltoallv.c src/page_rank_distributed.cc -lnuma

bfs_distributed: $(HEADERS) src/bfs_distributed.cc
	mpicxx -g -Wall -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o bfs_distributed src/util.cc src/ympi_alltoallv.c src/bfs_distributed.cc -lnuma

kmeans: $(HEADERS) src/kmeans.cc
	mpicxx -g -Wall -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o kmeans src/util.cc src/ympi_alltoallv.c src/kmeans.cc -lnuma


.PHONY: all clean
