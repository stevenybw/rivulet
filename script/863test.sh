SHM_PREFIX=/mnt/shm/user863
export NVM_OFF_CACHE_POOL_DIR=/mnt/pmem0/pool
export NVM_OFF_CACHE_POOL_SIZE=103079215104
export NVM_ON_CACHE_POOL_DIR=/mnt/pmem1/pool
export NVM_ON_CACHE_POOL_SIZE=103079215104

# twitter-2010 graph
INPUT_GRAPH_PATH=~/Dataset/twitter-2010
OUTPUT_GRAPH_PATH=/mnt/pmem0/twitter-2010
UNBALANCED_GRAPH_PATH=/mnt/pmem0/twitter-2010-unbalanced

# uk-2007-05 graph
INPUT_GRAPH_PATH_UK=~/Dataset/uk-2007-05
OUTPUT_GRAPH_PATH_UK=/mnt/pmem0/uk-2007-05

LOG=./result
HOSTS="phoenix00,phoenix01"
NUM_ITERS=10
CHUNK_SIZE=512


mkdir ${LOG}

function clear_result() {
  clush -w ${HOSTS} rm ${NVM_OFF_CACHE_POOL_DIR-"X"}/*;
  clush -w ${HOSTS} rm ${NVM_ON_CACHE_POOL_DIR-"X"}/*;
}

echo "6.1.1.1 数据对象模型跨节点"

echo "一个进程graph_preprocess"
mpirun --bind-to socket -n 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH}_chunk256k chunkroundrobin 262144

echo "两个进程graph_preprocess"
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH}_chunk256k chunkroundrobin 262144

echo "3.2.2.2 数据对象模型跨存储层次"

echo "存入共享内存"
export NVM_OFF_CACHE_POOL_DIR=/mnt/shm/pool
export NVM_OFF_CACHE_POOL_SIZE=68719476736
export NVM_ON_CACHE_POOL_DIR=/mnt/shm/pool
export NVM_ON_CACHE_POOL_SIZE=68719476736
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} /mnt/shm/twitter-2010_chunk256k chunkroundrobin 262144
clush -w phoenix0[0-1] -- "md5sum /mnt/shm/twitter-2010_chunk256k*.2"
clush -w phoenix0[0-1] -- "md5sum ${OUTPUT_GRAPH_PATH}_chunk256k*.2"

echo "3.2.2.3 数据对象模型可容错"

# 数据对象模型可容错
echo "重启系统，验证结果"

# 图计算
echo "预处理/twitter-2010/2*2进程"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH}_chunk256k chunkroundrobin 262144
echo "图领域编程范式/PageRank/twitter-2010"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE}
echo "图领域编程范式/BFS/twitter-2010"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./bfs_distributed ${OUTPUT_GRAPH_PATH}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE} 0 0

echo "预处理/uk-2007-05/2*2进程"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH_UK} ${OUTPUT_GRAPH_PATH_UK}_chunk256k chunkroundrobin 262144
echo "图领域编程范式/PageRank/uk-2007-05"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH_UK}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE}
echo "图领域编程范式/BFS/uk-2007-05"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./bfs_distributed ${OUTPUT_GRAPH_PATH_UK}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE} 0 0


# 流处理

echo "流处理领域编程范式/StreamingSum"
mpirun --bind-to none -npernode 1 --hostfile hostfile.txt -- ./stream_sum __random__ output.txt 1.25
echo "流处理领域编程范式/生成数据文件(1000000*16个单词)"
python script/gen_randword.py 1000000 > data/90m_text.txt
echo "流处理领域编程范式/StreamingWordCount"
mpirun --bind-to socket -npersocket 2 --hostfile hostfile.txt -- ./stream_wordcount /mnt/pmem0/staging/ output.txt
echo "拷贝数据"
mpirun --bind-to none -npernode 1 --hostfile hostfile.txt -- ./script/copy_files.sh

# 分布式资源调度框架

echo "分布式资源调度框架/MPI-YARN/1"
yarn jar ~/mpi-yarn-1.0.0.jar ai.fma.mpi_yarn.Client -a ./bin/cg.B.16 -j ~/mpi-yarn-1.0.0.jar -p hdfs://phoenix00/user/ybw/staging -lt orte -dp /usr/bin/orted -lp /usr/bin/orterun -N 2 -ppn 8 -vcpn 20 -mbmpn 65536 -mbnpn 65536 -q default -mbam 2048
echo "分布式资源调度框架/MPI-YARN/2"
yarn jar ~/mpi-yarn-1.0.0.jar ai.fma.mpi_yarn.Client -a ./busy.py -j ~/mpi-yarn-1.0.0.jar -p hdfs://phoenix00/user/ybw/staging -lt orte -dp /usr/bin/orted -lp /usr/bin/orterun -N 2 -ppn 8 -vcpn 20 -mbmpn 65536 -mbnpn 200000 -q default -mbam 2048
yarn application -status ${应用的APPID}
yarn application -kill ${busy.py的APPID}

# 负载均衡

echo "负载均衡/预处理equalvertex"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH}_equal_vertex equal_vertex

echo "负载均衡/点划分+静态调度"
make page_rank_distributed_static_scheduling
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed_static_scheduling ${OUTPUT_GRAPH_PATH}_equal_vertex ${NUM_ITERS} ${CHUNK_SIZE}

echo "负载均衡/块划分+静态调度"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed_static_scheduling ${OUTPUT_GRAPH_PATH}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE}

echo "负载均衡/点划分+动态调度"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH}_equal_vertex ${NUM_ITERS} ${CHUNK_SIZE}

echo "负载均衡/块划分+动态调度"
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE}

# CPU亲和性

echo "CPU亲和性/关闭CPU亲和性"
mpirun --bind-to none -npersocket 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH}_chunk256k ${NUM_ITERS} ${CHUNK_SIZE}

mpirun --bind-to none -npernode 1 --hostfile hostfile.txt -- ./stream_wordcount /mnt/pmem0/staging/ output.txt

# 性能测试/热词分析/延迟

echo "性能测试/热词分析/延迟/数据生成"
python script/gen_randword.py 4096 > data/small_text.txt
echo "性能测试/热词分析/延迟/StreamingWordCount"
mpirun --bind-to socket -npersocket 2 --hostfile hostfile.txt -- ./stream_wordcount /mnt/pmem0/staging/ output.txt 10000 2>&1 | tee result.txt
echo "性能测试/热词分析/延迟/拷贝数据"
./script/copy_file_latency.sh
echo "分析StreamingWordCount结果"
cat result.txt | grep -e Timestamp -e "Read" | grep "Read from" -B 1 | grep "Timestamp"

echo "性能测试/热词分析/带宽/StreamingWordCount"
mpirun --bind-to socket -npersocket 2 --hostfile hostfile.txt -- ./stream_wordcount /mnt/pmem0/staging/ output.txt 10000 2>&1 | tee result.txt
echo "性能测试/热词分析/带宽/拷贝数据"
mpirun --bind-to none -npernode 1 --hostfile hostfile.txt -- ./script/copy_files.sh


echo "性能测试/网页排名"




mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./stream_sum __random__ output.txt
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./stream_wordcount __random__ output.txt

mpirun --bind-to none -n 2 -host ${HOSTS} ./page_rank_distributed ${OUTPUT_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE}
mpirun -report-bindings -bind-to socket -npersocket 1 -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE}
mpirun --bind-to none -n 2 -host ${HOSTS} ./page_rank_distributed ${OUTPUT_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE}

# Four Process PageRank
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./page_rank_distributed ${OUTPUT_GRAPH_PATH} 4 ${CHUNK_SIZE}
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./page_rank_distributed ${UNBALANCED_GRAPH_PATH} 4 ${CHUNK_SIZE}

# Balanced 4 process
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt ./page_rank_distributed ${OUTPUT_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE}

# Unbalanced 4 process
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt ./page_rank_distributed ${UNBALANCED_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE}

# BFS从0号顶点开始
mpirun --bind-to none -n 2 -host ${HOSTS} ./bfs_distributed ${OUTPUT_GRAPH_PATH} ${NUM_ITERS} ${CHUNK_SIZE} 0 0

# 流处理
mpirun --bind-to none -n 2 -host phoenix00,phoenix01 ./stream_wordcount __random__ output.txt
mpirun --bind-to none -n 2 -host phoenix00,phoenix01 ./stream_wordcount /mnt/shm output.txt

# 机器学习？

#负载均衡

# 2 Process
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${UNBALANCED_GRAPH_PATH} equal_vertex
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} chunkroundrobin
mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH}_chunk4k chunkroundrobin 4096

mpirun --bind-to socket -npernode 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} chunkroundrobin
mpirun --bind-to socket -npersocket 1 --hostfile hostfile.txt -- ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} chunkroundrobin