# 2.85508s
OMP_NUM_THREADS=18 numactl --membind=0 --cpunodebind=0 -- /usr/bin/time -v -- ./page_rank /export/ybw/twitter-2010/twitter-2010-csr /export/ybw/twitter-2010/twitter-2010-t-csr push 10 16 64

# 1.66526s
OMP_NUM_THREADS=18 numactl --membind=0 --cpunodebind=0 -- /usr/bin/time -v -- ./page_rank /export/ybw/twitter-2010/twitter-2010-csr /export/ybw/twitter-2010/twitter-2010-t-csr delegation_pwq 10 16 64
