HEADERS := include/rivulet.h include/rivulet_impl.h include/channel.h

page_rank: $(HEADERS) src/page_rank.cc
	mpicxx -g -O2 -Iinclude -lnuma -std=c++14 -DUPDATE_CAS -march=native -fopenmp -o page_rank src/util.cc src/page_rank.cc

#mpicxx -g -O0 -Iinclude -lnuma -std=c++14 src/stream_sum.cc

