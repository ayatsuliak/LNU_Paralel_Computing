#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include <time.h>
#include <chrono>
#include <algorithm>
#include "matrix.h"
#include "matrix.cpp"

using namespace std;

void fillMatrix(Matrix <int>& mtrx);

template <typename T>
void printMatrix(Matrix <T>& mtrx);

template <typename T>
void multiplicateSequentically(Matrix <T>& mtrxA, Matrix <T>& mtrxB, Matrix <T>& mtrxRes);

#define ROW_START_TAG 0   
#define ROW_END_TAG 1     
#define A_ROWS_TAG 2      
#define C_ROWS_TAG 3       
#define LOCAL_TIME_TAG 4   

int nProcesses;
MPI_Status status;
MPI_Request request;
size_t rowStart, rowEnd;
size_t granularity;

double start_time, end_time;
double localTimeSaver;

int main(int argc, char* argv[])
{
    srand(time(NULL));

    int rank;

    if (argv[1] == NULL) {
        return 1;
    }

    int N = atoi(argv[1]);

    int numberOfRowsA = N;
    int numberOfColsA = N;
    int numberOfRowsB = N;
    int numberOfColsB = N;

    Matrix <int> A = Matrix <int>(numberOfRowsA, numberOfColsA);
    Matrix <int> B = Matrix <int>(numberOfRowsB, numberOfColsB);
    Matrix <int> seqC = Matrix <int>(numberOfRowsA, numberOfColsB);
    Matrix <int> mpiC = Matrix <int>(numberOfRowsA, numberOfColsB);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nProcesses);

    if (rank == 0)
    {
        cout << "The matrices are: " << N << "x" << N << endl;
        fillMatrix(A);
        fillMatrix(B);

        start_time = MPI_Wtime();
        for (int i = 1; i < nProcesses; i++)
        {
            granularity = (numberOfRowsA / (nProcesses - 1));
            rowStart = (i - 1) * granularity;

            if (((i + 1) == nProcesses) && ((numberOfRowsA % (nProcesses - 1)) != 0)) 
            {
                rowEnd = numberOfRowsA; 
            }
            else
            {
                rowEnd = rowStart + granularity; 
            }

            MPI_Isend(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&A(rowStart, 0), (rowEnd - rowStart) * numberOfColsA, MPI_INT, i, A_ROWS_TAG, MPI_COMM_WORLD, &request);
        }
    }
    MPI_Bcast(&B(0, 0), numberOfRowsB * numberOfColsB, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank > 0)
    {
        MPI_Recv(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&A(rowStart, 0), (rowEnd - rowStart) * numberOfColsA, MPI_INT, 0, A_ROWS_TAG, MPI_COMM_WORLD, &status);

        localTimeSaver = MPI_Wtime();

        for (int i = rowStart; i < rowEnd; i++)
        {
            for (int j = 0; j < B.getCols(); j++)
            {
                for (int k = 0; k < B.getRows(); k++)
                {
                    mpiC(i, j) += (A(i, k) * B(k, j));
                }
            }
        }
        localTimeSaver = MPI_Wtime() - localTimeSaver;

        MPI_Isend(&rowStart, 1, MPI_INT, 0, ROW_END_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&rowEnd, 1, MPI_INT, 0, ROW_START_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&mpiC(rowStart, 0), (rowEnd - rowStart) * numberOfColsB, MPI_INT, 0, C_ROWS_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&localTimeSaver, 1, MPI_INT, 0, LOCAL_TIME_TAG, MPI_COMM_WORLD, &request);
    }

    if (rank == 0)
    {
        for (int i = 1; i < nProcesses; i++)
        {
            MPI_Recv(&rowStart, 1, MPI_INT, i, ROW_END_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&rowEnd, 1, MPI_INT, i, ROW_START_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&mpiC(rowStart, 0), (rowEnd - rowStart) * numberOfColsB, MPI_INT, i, C_ROWS_TAG, MPI_COMM_WORLD, &status);
        }
        end_time = MPI_Wtime();
        double totalMultiplicationTime = end_time - start_time;

        vector<double> LocalMultiplicationTimes = vector<double>(nProcesses); 

        for (int i = 1; i < nProcesses; i++)
        {
            MPI_Recv(&LocalMultiplicationTimes[i], 1, MPI_INT, i, LOCAL_TIME_TAG, MPI_COMM_WORLD, &status);
        }

        cout << "Total mpi time =  " << totalMultiplicationTime << endl;

        auto seqStart = chrono::high_resolution_clock::now();
        multiplicateSequentically(A, B, seqC);
        auto seqEnd = chrono::high_resolution_clock::now();
        chrono::duration<double> seqDuration = seqEnd - seqStart;
        cout << "Sequential multiplication time = " << seqDuration.count() << " seconds\n";

        if (N <= 10)
        {
            cout << "Matrix A:\n";
            printMatrix(A);
            cout << "Matrix B:\n";
            printMatrix(B);
            cout << "Matrix C(mpi):\n";
            printMatrix(mpiC);
            cout << "Matrix C(seq):\n";
            printMatrix(seqC);
        }
    }
    MPI_Finalize();

    return 0;
}

void fillMatrix(Matrix <int>& mtrx)
{
    for (size_t i = 0; i < mtrx.getRows(); i++)
    {
        for (size_t j = 0; j < mtrx.getCols(); j++)
        {
            mtrx(i, j) = rand() % 100;
        }
    }
}

template <typename T>
void printMatrix(Matrix <T>& mtrx)
{
    for (size_t i = 0; i < mtrx.getRows(); i++)
    {
        for (size_t j = 0; j < mtrx.getCols(); j++)
        {
            cout << mtrx(i, j) << " ";
        }
        cout << endl;
    }
}

template <typename T>
void multiplicateSequentically(Matrix <T>& mtrxA, Matrix <T>& mtrxB, Matrix <T>& mtrxRes)
{
    size_t rowsA = mtrxA.getRows();
    size_t colsA = mtrxA.getCols();
    size_t colsB = mtrxB.getCols();

    for (size_t i = 0; i < rowsA; i++)
    {
        for (size_t j = 0; j < colsB; j++)
        {
            for (size_t k = 0; k < colsA; k++)
            {
                mtrxRes(i, j) += mtrxA(i, k) * mtrxB(k, j);
            }
        }
    }
}