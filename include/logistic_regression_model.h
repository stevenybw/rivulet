#ifndef RIVULET_LOGISTIC_REGRESSION_MODEL_H
#define RIVULET_LOGISTIC_REGRESSION_MODEL_H

#include <mpi.h>
#include <omp.h>
#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include <random>
#include "garray.h"
#include "driver.h"

#define DELETE_ARRAY(pData) { if(pData!=nullptr){delete []pData; pData=nullptr;} }

struct LogisticRegressionModel{
    uint64_t duration;
    ExecutionContext& ctx;
    MPI_Comm comm;
    int rank;
    int nprocs;

    float* x;
    int* y;
    float* w = nullptr;
    float ratio;
    long long num;
    long long dim;

    LogisticRegressionModel(ExecutionContext& ctx, uint64_t _dur)
            :ctx(ctx), comm(ctx.get_comm()), rank(ctx.get_rank()), nprocs(ctx.get_nprocs())
    {
		duration = _dur;
    }
    ~LogisticRegressionModel(){
		DELETE_ARRAY(w);
    }

    void train(GArray<float>* data, GArray<int>* label, long long _dim, float _ratio, int ite){
        assert(data->_size % _dim == 0);
        assert(data->_size / _dim == label->_size);
        x = data->data();
        y = label->data();
        num = data->_size / _dim;
        dim = _dim;
        ratio = _ratio;

        w = new float[dim];
        memset(w, 0, dim*sizeof(float));
        float *t = new float[num];
        float *d = new float[dim];
        for(int i=0; i<ite; i++){
			char info[100];
			sprintf(info, "Iteration: %d", i);
			LOG_INFO(info);
			iteration(t, d);
        }
		DELETE_ARRAY(t);
        DELETE_ARRAY(d);
    }

    void iteration(float *t, float *d){
        memset(t, 0, num * sizeof(float));
        memset(d, 0, dim * sizeof(float));

#pragma omp parallel for
        for(long long i=0; i<num; i++){
            float tmp = 0.0;
#pragma omp simd reduction(+:tmp)
            for(long long m=0; m<dim; m++){
                tmp += (x[i*dim+m] * w[m]);
            }
            tmp = 1.0/(1.0 + exp(-tmp));
            t[i] = tmp - y[i];
        }

#pragma omp parallel for
        for(long long m=0; m<dim; m++){
            float tmp = 0.0;
            for(long long i=0; i<num; i++){
                tmp += (t[i]*x[i*dim+m]);
            }
            d[m] = ratio*tmp;
        }

        MPI_Allreduce(MPI_IN_PLACE, d, dim, MPI_FLOAT, MPI_SUM, comm);
#pragma omp simd
        for(long long m=0; m<dim; m++){
            w[m] -= d[m];
        }

    }

    float* getWeight(){
        return w;
    }

};

struct LogisticRegressionPredictModel{
    uint64_t duration;
    ExecutionContext& ctx;
    MPI_Comm comm;
    int rank;
    int nprocs;

    long long num;
    long long dim;

    float *w;
    float *x;
    int *y;

    int *pre_labels;
    float accuracy = -1;
    long long right_num;

    LogisticRegressionPredictModel(ExecutionContext& ctx)
            :ctx(ctx), comm(ctx.get_comm()), rank(ctx.get_rank()), nprocs(ctx.get_nprocs())
    {}
    ~LogisticRegressionPredictModel(){
        DELETE_ARRAY(pre_labels);
    }

    void predict(GArray<float>* data, float *_w, long long _dim, GArray<int>* labels = nullptr){
        assert(data->_size % _dim == 0);
        num = data->_size / _dim;
        dim = _dim;
        x = data->data();
        if(labels != nullptr)
            y = labels->data();
        w = _w;
        pre_labels = new int[num];
        memset(pre_labels, 0, num* sizeof(int));

        long long count = 0;
#pragma omp parallel for reduction(+:count)
        for(long long i=0; i<num; i++){
            float z = 0.0;
#pragma omp simd reduction(+:z)
            for(long long m=0; m<dim; m++){
                z += (x[i*dim+m] * w[m]);
            }
            z = 1.0/(1.0 + exp(-z));
            pre_labels[i] = z>=0.5? 1 : 0;
            if(labels != nullptr && pre_labels[i] == y[i]){
                count += 1;
            }
        }
		right_num = count;
        MPI_Allreduce(MPI_IN_PLACE, &right_num, 1, MPI_LONG_LONG, MPI_SUM, comm);
		long long total;
		MPI_Allreduce(&num, &total, 1, MPI_LONG_LONG, MPI_SUM, comm);
        if(labels != NULL)
            accuracy = float(right_num)/total;
    }

    int *getLabels(){
        return pre_labels;
    }


};


#endif //RIVULET_LOGISTIC_REGRESSION_MODEL_H
