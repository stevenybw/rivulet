#ifndef COMMON_H
#define COMMON_H

#include <assert.h>

// #define PRINTF(fmt, ...) printf(fmt, __VA_ARGS__)
#define PRINTF(fmt, ...)
#define LINES do{int rank; MPI_Comm_rank(MPI_COMM_WORLD, &rank); if(rank == 0) printf("  %s:%d\n", __FUNCTION__, __LINE__); }while(0)
#define REGION_BEGIN() do{duration = -currentTimeUs();}while(0)
#define REGION_END(msg) do{duration += currentTimeUs(); if (rank == 0){printf("  [%s] %s Time: %lf sec\n", __FUNCTION__, msg, 1e-6 * duration);}}while(0)
#define LOG_BEGIN() do{duration = -currentTimeUs();}while(0)
#define LOG_INFO(msg) do{if(rank == 0) {printf("[INFO-%d@%s@%lf] %s\n", g_rank, __FUNCTION__, (1e-6 * (duration + currentTimeUs())), msg);}}while(0)

#endif
