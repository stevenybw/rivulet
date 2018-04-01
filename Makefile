EXECUTABLES := stream_wordcount stream_sum page_rank

HEADERS := include/rivulet.h include/rivulet_impl.h include/channel.h

all: $(EXECUTABLES)

clean:
	rm $(EXECUTABLES)

stream_wordcount: $(HEADERS) src/stream_wordcount.cc
	mpicxx -g -O2 -Iinclude -std=c++14 -march=native -fopenmp -o stream_wordcount src/util.cc src/stream_wordcount.cc -lnuma

stream_sum: $(HEADERS) src/stream_sum.cc
	mpicxx -g -O2 -Iinclude -std=c++14 -march=native -fopenmp -o stream_sum src/util.cc src/stream_sum.cc -lnuma

page_rank: $(HEADERS) src/page_rank.cc
	mpicxx -g -O2 -Iinclude -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o page_rank src/util.cc src/page_rank.cc -lnuma

.PHONY: all clean