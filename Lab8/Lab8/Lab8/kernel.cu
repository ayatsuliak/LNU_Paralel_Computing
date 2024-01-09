#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "kernel.h"
#include <stdlib.h>
using namespace std;

__global__ void matrixMultiplicationKernel(float* A, float* B, float* C, int N)
{
    int ROW = blockIdx.y * blockDim.y + threadIdx.y;
    int COL = blockIdx.x * blockDim.x + threadIdx.x;

    float tmpSum = 0;

    if (ROW < N && COL < N)
    {
        for (int i = 0; i < N; i++)
        {
            tmpSum += A[ROW * N + i] * B[i * N + COL];
        }
        C[ROW * N + COL] = tmpSum;
    }
}


void matrixMultiplication(float* A, float* B, float* C, int N, int threadsPerBlock)
{
    dim3 threads(threadsPerBlock, threadsPerBlock);
    dim3 blocks((N + threads.x - 1) / threads.x, (N + threads.y - 1) / threads.y);

    matrixMultiplicationKernel < << blocks, threads >> > (A, B, C, N);
}