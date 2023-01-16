#ifndef _CUDA_UTILS_H
#define _CUDA_UTILS_H

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

#define TOTAL_THREADS 512

inline int opt_n_thread(int work_size){
    const int pow_2 = std::log(static_cast<double>(work_size)) / std::log(2.0);
    return max(min(1<<pow_2, TOTAL_THREADS), 1);
}

inline dim3 opt_block_config(int x, int y){
    const int x_thread = opt_n_thread(x);
    const int y_thread = max(min(opt_n_thread(y), TOTAL_THREADS/x_thread), 1);
    dim3 block_config(x_thread, y_thread, 1);

    return block_config;
}

# define CUDA_CHECK_ERRORS()                                                \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (cudaSuccess!=err){                                              \
            fprintf(stderr, "CUDA kernel failed : %s\n%s at L:%d in %s\n",  \
                cudaGetErrorString(err), __PRETTY_FUNCTION__, __LINE__,     \
                __FILE__);                                                  \
            exit(-1);                                                       \
        }                                                                   \
    } while(0)                                                              \

#endif