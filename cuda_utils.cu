#include <stdio.h>
#include <stdlib.h>
#include "cuda.h"
#include <cmath>

#define CUDA_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

#define N 1024



// Kernel multiply two matrices
/*
matrix1 (nxp)
matrix2 (mxp)
OutputMatrix (nxm)

block and grid configuration:
one block (p,1,1) for each vector multiplication then will be one block for each every element in the OutputMatrix
so the grid dimension will be (n, m, 1) 
*/
__global__ void easymatmult(float *matrix1, float *matrix2, float *OutputMatrix, int n, int p, int m){

    int dim_mat1 = blockIdx.x * p + threadIdx.x;
    int dim_mat2 = blockIdx.y * p + threadIdx.x;

    if threadIdx.x<p and dim_mat1<n*p and dim_mat2<m*p{
        OutputMatrix[dim_mat1 * m + dim_mat2].attomicAdd(matrix1[dim_mat1]*matrix2[dim_mat2]);
    } 

    __syncthreads();

}


// Kernel multiply a matrix with a vector
/*
matrix (nxp)
vector (px1) 
OutputVector (nx1)

block and grid configuration:
one block (p,1,1) for each vector multiplication then will be one block for each every element in the OutputVector
so the grid dimension will be (n, 1, 1) 
*/
__global__ void easyvectmult(float *matrix, float *vector, int *OutputVector, int n, int p){

    int dim_mat1 = blockIdx.x * p + threadIdx.x;
    if threadIdx.x < p {
        OutputVector[blockIdx.x].attomicAdd(matrix[dim_mat1]*vector[threadIdx.x])
    }

    __syncthreads();

}


// Kernel to sum a int value to a vector 
/*
In order to make the stack faster, I will apply this sum to a matrix and the value
will be a vector. The vectors will be the rows of the matrix to make it faster.
vector (pxn)
values (nx1)

Each block will be assigend to each row of the matrix.
*/
__global__ void easyvectsum(float *matrix, float *vajlues, int p, int n) {
    dim_matrix = blockIdx.x*p + threadIdx.x;
    if threadIdx.x < p {
        matrix[dim_matrix]+=values[blockDim.x];
    }
}


//Kernel to apply a gelu function to every element in a vector
/*
vector (nxp)
This function will apply the gelu function to every element of a matrix
*/
__global__ void RELU(float *matrix, int p ){
    dim_matrix = blockIdx.x*p + threadIdx.x;
    if threadIdx.x < p{
        matrix[dim_matrix] = matrix[dim_matrix] > 0 ? matrix[dim_matrix] : 0;
    }
}

//Kernel to normalize each row of a matrix
/*
matrix(nxp)

*/
__global__ void matrixNorm(float *matrix, int p){
    dim_matrix = blockIdx.x*p + threadIdx.x;
    __shared__ int total;
    if threadIdx.x<p{
        total.attomicAdd(matrix[dim_matrix]*matrix[dim_matrix]);
    }

    __syncthreads();

    if threadIdx.x<p{
        matrix[dim_matrix]=matrix[dim_matrix]/sqrt(total)
    }
}

//Kernel to transpose a matrix
/*
inputMatrix (nxm)
outputMatrix (mxn)
*/
__global__ void transposeMatrix(int *inputMatrix, int *outputMatrix, int n, int m) {

    __shared__ int shared_tile[32 * 32];

    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    if row < n && col < m{
        shared_tile [threadIdx.x * 32 + threadIdx.y] = inputMatrix [col * n + row];
    }
    __syncthreads();
    if row < n && col < m{
        int row_output = blockIdx.y * blockDim.y + threadIdx.x;
        int col_output = blockIdx.x * blockDim.x + threadIdx.y;

        outputMatrix[col_output * n + row_output] = shared_tile[threadIdx.y * 32 + threadIdx.x];
    }
}








