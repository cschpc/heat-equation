#pragma once
// This header provides two helper macros for error checking

#include <cstdio>
#include <cstdlib>

#define GPU_CHECK(errarg)   __checkErrorFunc(errarg, __FILE__, __LINE__)
#define CHECK_ERROR_MSG(errstr) __checkErrMsgFunc(errstr, __FILE__, __LINE__)

inline void __checkErrorFunc(cudaError_t errarg, const char* file, 
                             const int line)
{
    if(errarg != cudaSuccess) {
        fprintf(stderr, "Error at %s(%i): %s)\n", file, line, cudaGetErrorString(errarg));
        exit(EXIT_FAILURE);
    }
}


inline void __checkErrMsgFunc(const char* errstr, const char* file, 
                              const int line)
{
    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess) {
        fprintf(stderr, "Error: %s at %s(%i): %s\n", 
                errstr, file, line, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

