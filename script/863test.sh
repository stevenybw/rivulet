SHM_PREFIX=/mnt/shm/user863
export NVM_OFF_CACHE_POOL_DIR=/mnt/pmem0/pool
export NVM_OFF_CACHE_POOL_SIZE=103079215104
export NVM_ON_CACHE_POOL_DIR=/mnt/pmem1/pool
export NVM_ON_CACHE_POOL_SIZE=103079215104

INPUT_GRAPH_PATH=~/Dataset/twitter-2010
OUTPUT_GRAPH_PATH=${NVM_OFF_CACHE_POOL_DIR}/twitter-2010
LOG=./result
HOSTS="phoenix00,phoenix01"
NUM_ITERS=10
CHUNK_SIZE=512


mkdir ${LOG}

function clear_result() {
  clush -w ${HOSTS} rm ${NVM_OFF_CACHE_POOL_DIR-"X"}/*;
  clush -w ${HOSTS} rm ${NVM_ON_CACHE_POOL_DIR-"X"}/*;
}

echo "3.2.2.1 数据对象模型跨节点"
clear_result
echo "一个进程graph_preprocess"
mpirun -n 1 ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} 2>&1 | tee ${LOG}/3.2.2.1_one_process.txt
clear_result
echo "两个进程graph_preprocess"
mpirun -n 2 --host ${HOSTS} ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} 2>&1 | tee ${LOG}/3.2.2.1_two_process.txt

echo "3.2.2.2 数据对象模型跨存储层次"
clear_result
echo "存入共享内存"
mpirun -n 2 --host ${HOSTS} ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} ${SHM_PREFIX}
md5sum ${SHM_PREFIX}/* 2>&1 | tee ${LOG}/shm_md5.txt
echo "存入NVDIMM"
mpirun -n 2 --host ${HOSTS} ./graph_preprocess ${INPUT_GRAPH_PATH} ${OUTPUT_GRAPH_PATH} ${NVM_PREFIX}
md5sum ${NVM_PREFIX}/* 2>&1 | tee ${LOG}/nvm_md5.txt

echo "3.2.2.3 数据对象模型可容错"

# 数据对象模型可容错
echo "重启系统，验证结果"

# 用户控制的重新排布

# 流处理
mpirun --bind-to none -n 2 -host phoenix00,phoenix01 ./stream_wordcount __random__ output.txt
mpirun --bind-to none -n 2 -host phoenix00,phoenix01 ./stream_wordcount /mnt/shm output.txt