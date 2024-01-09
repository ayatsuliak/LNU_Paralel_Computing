#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"
#include "kernel.h"
#include "dev_array.cpp"
#include <math.h>
#include <time.h>
#include <chrono>

using namespace std;

void printMatrix(const float* matrix, int rows, int cols)
{
    for (size_t i = 0; i < rows; i++)
    {
        for (size_t j = 0; j < cols; j++)
        {
            cout << matrix[i * rows + j] << " ";
        }
        cout << endl;
    }
}

int main()
{
    srand(time(NULL));
    int N = 1000;
    int SIZE = N * N;
    const int THREADS = 32;

    vector<float> h_A(SIZE);
    vector<float> h_B(SIZE);
    vector<float> h_C(SIZE);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            h_A[i * N + j] = rand() % 10 + 1;
            h_B[i * N + j] = rand() % 10 + 1;
        }
    }

    /*cout << "Matrix A:" << endl;
    //printMatrix(&h_A[0], N, N);
    //cout << "Matrix B:" << endl;
    printMatrix(&h_B[0], N, N);*/

    dev_array<float> d_A(SIZE);
    dev_array<float> d_B(SIZE);
    dev_array<float> d_C(SIZE);

    d_A.set(&h_A[0], SIZE);
    d_B.set(&h_B[0], SIZE);

    auto start = chrono::high_resolution_clock::now();

    matrixMultiplication(d_A.getData(), d_B.getData(), d_C.getData(), N, THREADS);
    cudaDeviceSynchronize();

    auto end = chrono::high_resolution_clock::now();
    auto duration = chrono::duration_cast<chrono::microseconds>(end - start);

    cudaError_t cudaError = cudaGetLastError();
    if (cudaError != cudaSuccess)
    {
        cout << "CUDA Error: " << cudaGetErrorString(cudaError) << endl;
    }

    cout << "Parallel multiplication for " << THREADS << " threads: " << duration.count() << " msec\n";

    d_C.get(&h_C[0], SIZE);
    cudaDeviceSynchronize();

    /*cout << "Matrix C (GPU):" << endl;
    printMatrix(&h_C[0], N, N);*/

    float* cpu_C;
    cpu_C = new float[SIZE];

    start = chrono::high_resolution_clock::now();
    float sum;
    for (int row = 0; row < N; row++)
    {
        for (int col = 0; col < N; col++)
        {
            sum = 0.f;
            for (int n = 0; n < N; n++)
            {
                sum += h_A[row * N + n] * h_B[n * N + col];
            }
            cpu_C[row * N + col] = sum;
        }
    }
    end = chrono::high_resolution_clock::now();
    duration = chrono::duration_cast<chrono::microseconds>(end - start);
    cout << "Sequentical multiplication: " << duration.count() << " msec\n";

    /*cout << "Matrix C (CPU):" << endl;
    printMatrix(cpu_C, N, N);*/

    double err = 0;
    for (int ROW = 0; ROW < N; ROW++)
    {
        for (int COL = 0; COL < N; COL++)

        {
            err += cpu_C[ROW * N + COL] - h_C[ROW * N + COL];
        }
    }

    cout << "Error: " << err << endl;

    delete[] cpu_C;

    return 0;
}
