#include "matrix.h"
#include <mpi.h>
#include <iostream>

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    // ��������� ����� �� ������� �������
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = atoi(argv[1]);

    // �������� ���
    Matrix matrixA(N, N);
    Matrix matrixB(N, N);

    double startTime, endTime;

    if (rank == 0) {
        // ���� ������ 0 ������� �� �������� �������
        matrixA.fillMatrix();
        matrixB.fillMatrix();
    }

    // �������� ������� B �� ��� �������
    MPI_Bcast(&(matrixB(0, 0)), matrixB.getRows() * matrixB.getColumns(), MPI_INT, 0, MPI_COMM_WORLD);

    // �������� ��� ��� ������������ ��������
    MPI_Barrier(MPI_COMM_WORLD);
    startTime = MPI_Wtime();

    Matrix resultMatrixParallelPart = matrixA.multiplyMatrixByParallelPart(matrixB, rank, size);

    MPI_Barrier(MPI_COMM_WORLD);
    endTime = MPI_Wtime();

    // ��� ���������� �� ������ 0
    if (rank == 0) {
        Matrix resultMatrixParallel = matrixA.createEmptyMatrix();
        MPI_Gather(resultMatrixParallelPart.getDataPointer(), resultMatrixParallelPart.getRows() * resultMatrixParallelPart.getColumns(), MPI_INT,
            resultMatrixParallel.getDataPointer(), resultMatrixParallel.getRows() * resultMatrixParallel.getColumns(), MPI_INT,
            0, MPI_COMM_WORLD);

        // ���� ���������� ����� ��� ������� 0
        std::cout << "Matrix A:" << std::endl;
        matrixA.printMatrix();
        std::cout << "Matrix B:" << std::endl;
        matrixB.printMatrix();
        std::cout << "Result Matrix (Parallel):" << std::endl;
        resultMatrixParallel.printMatrix();

        // ���� ���� ��������� ������������ ��������
        std::cout << "Parallel Execution Time: " << endTime - startTime << " seconds." << std::endl;
    }
    else {
        // ���� ���� ��������� ��� ����� �������
        std::cout << "Process " << rank << " Execution Time: " << endTime - startTime << " seconds." << std::endl;
    }

    MPI_Finalize();
    return 0;
}
