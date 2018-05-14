#ifndef RIVULET_KMEANS_MODEL_H
#define RIVULET_KMEANS_MODEL_H

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <stdlib.h>
#include <random>
#include "garray.h"
#include "driver.h"

#define DELETE_ARRAY(pData) { if(pData!=nullptr){delete []pData; pData=nullptr;} }

struct KMeansModel{
	uint64_t duration;
    ExecutionContext& ctx;
    MPI_Comm comm;
    int rank;
    int nprocs;

    float* data = nullptr;
    int num; // number of samples in this process
    int dim; // dimension of samples
    int k;   // number of clusters
    int ite; // max of iterations

    float* centers = nullptr;
    int* mark = nullptr; // count[i] : num of samples belonging to cluster[i]
    int* count = nullptr; // mark[i] : cluster that sample[i] belongs to
    float cost = -1;

    KMeansModel(ExecutionContext& ctx):ctx(ctx), comm(ctx.get_comm()), rank(ctx.get_rank()), nprocs(ctx.get_nprocs()){
	LOG_BEGIN();
	}
    ~KMeansModel(){
        DELETE_ARRAY(centers);
		DELETE_ARRAY(mark);
		DELETE_ARRAY(count);
    }


	void print(float* x, int row, int col){
		for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
				printf("%f ", x[i*col+j]);	
			}
			printf("\n");
		}
	}


	void print(int* x, int row, int col){
		for(int i=0; i<row; i++){
			for(int j=0; j<col; j++){
				printf("%d ", x[i*col+j]);	
			}
			printf("\n");
		}
	}

    void train(GArray<float>* garray, int _dim, int _k, int _ite){
        assert(garray->_size % _dim == 0); // ensure complete sample
        data = garray->data();
        num = garray->_size / _dim;
        dim = _dim;
        k = _k;
        ite = _ite;
        initCenters();
        count = new int[k];
        mark = new int[num];
        for(int i=0; i<ite; i++){
            char info[100];
            sprintf(info, "Iteration: %d", i);
			LOG_INFO(info);
            iteration();
            sync();
		//	float cost = computeCost();	
		//	if(rank==0){
			//	print(centers, k, dim);
			//	print(mark, 1, num);
			//	print(count, 1, k);
		//		printf("cost after sync: %f \n", cost);
		//	}
		}
    }

    void iteration(){
		memset(count, 0, k*sizeof(float));
#pragma omp parallel for
        for(int i=0; i<num; i++){
            float dist[k] = {0}; // distance from every center
            float min = 1e20;
            int min_idx =-1;
            for(int j=0; j<k; j++){
                float delta[dim];
                // TODO: simd?
                for(int m=0; m<dim; m++){
                    delta[m] = data[i*dim+m] - centers[j*dim+m];
                    delta[m] = delta[m] * delta[m];
                    dist[j] += delta[m];
                }
                // norm
//                dist[j] = cblas_snrm2(dim, delta, 1);
                if(dist[j] < min){
                    min = dist[j];
                    min_idx = j;
                }
            }
//            mark[i] = (int)cblas_isamin(k, dist, 1);
            assert(min_idx != -1);
            mark[i] = min_idx;
            __sync_fetch_and_add(&count[mark[i]], 1);
			//count[mark[i]] += 1;
        }
        // update centers
        memset(centers, 0, k*dim*sizeof(float));
        // parallel by samples: modify centers critically
        // parallel by clusters: can not achieve load balance
        // so parallel by dimension: unfriendly to cache, but load balance
        // TODO: is there more efficient implementation?
#pragma omp parallel for
        for(int m=0; m<dim; m++){
            for(int i=0; i<num; i++){
                centers[ mark[i]*dim+m ] += data[i*dim+m];
            }
        }
    }

    void sync(){
        // AllReduce: centers, count,
//			if(rank==1 || rank == 0){
//				print(centers, k, dim);
//				print(mark, 1, num);
//				print(count, 1, k);
//				//printf("cost before sync: %f \n", computeCost());
//			}
        MPI_Allreduce(MPI_IN_PLACE, centers, k*dim, MPI_FLOAT, MPI_SUM, comm);
        MPI_Allreduce(MPI_IN_PLACE, count, k, MPI_INT, MPI_SUM, comm);
//			if(rank==1 || rank == 0){
//				print(centers, k, dim);
//				print(mark, 1, num);
//				print(count, 1, k);
//				//printf("cost before sync: %f \n", computeCost());
//			}
#pragma omp parallel for collapse(2)
        for(int i=0; i<k; i++){
            for(int j=0; j<dim; j++){
                if(count[i] != 0){
					centers[i*dim+j] /= count[i];
            	}
			}
        }
    }

    void initCenters(){
        LOG_INFO("Initialize centers randomly.");
        if(centers != nullptr){
            delete centers;
            centers = nullptr;
        }
        centers = new float[k * dim];
        // get total num
        int total = 0;
        MPI_Allreduce(&num, &total, 1, MPI_INT, MPI_SUM, comm);
        int indexes[k] = {0};
        if(rank == 0){
            srand((unsigned)time(0));
#pragma omp parallel
            for(int i=0; i<k; i++){
                indexes[i] = rand() % total;
            }
        }
		MPI_Bcast(indexes, k, MPI_INT, 0, comm);
//		print(indexes, 1, k);
        int chunk_size = total/nprocs;
        int from_idx = chunk_size*rank;
        int to_idx = (rank == (nprocs-1))?total:(rank+1)*chunk_size;
//		printf("rank: %d, from_idx: %d, to_idx: %d", rank, from_idx, to_idx);
#pragma omp parallel
        for(int i=0; i<k; i++){
            if(indexes[i] >= from_idx && indexes[i] < to_idx){
                // sample in this process
                int offset = (indexes[i]-from_idx)*dim;
                for(int j=0; j<dim; j++){
                    centers[i*dim + j] = data[offset+j];
                }
            }
        }
        LOG_INFO("Reduce centers to all processes.");
//	    print(centers, k, dim);	
        MPI_Allreduce(MPI_IN_PLACE, centers, k*dim, MPI_FLOAT, MPI_SUM, comm);
//        MPI_Bcast(centers, k*dim, MPI_FLOAT, 0, comm);
  //  	print(centers, k, dim);
	}

    float* getCenters(){
        return centers;
    }

    float computeCost(){
        if(data == nullptr || centers == nullptr)
            return -1;
        // within set sum of squared errors
        float sum = 0;
#pragma omp parallel for collapse(2) reduction(+:sum)
        for(int i=0; i<num; i++){
            for(int m=0; m<dim; m++){
                float squared_error = (data[i*dim+m]-centers[mark[i]*dim+m]);
                sum += (squared_error*squared_error);
            }
        }
        MPI_Allreduce(&sum, &cost, 1, MPI_FLOAT, MPI_SUM, comm);
        return cost;
    }

    void delPtr(void* p){
        if(p != nullptr){
            delete []p;
        }
    }
};


#endif //RIVULET_KMEANS_MODEL_H
