#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H


__global__ void easymatmult(float *matrix1, float *matrix2, float *OutputMatrix, int n, int p, int m);
__global__ void easyvectmult(float *matrix, float *vector, int *OutputVector, int n, int p);
__global__ void easyvectsum(float *matrix, float *values, int p, int n);
__global__ void GELU(float *matrix, int p );
__global__ void matrixNorm(float *matrix, int p);
__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix, int n);



#endif 
