#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <thread>
#include <vector>
#include <stdexcept>
#include <mutex>

class Matrix {
private:
    int rows;
    int columns;
    std::vector<std::vector<int>> matrix;
    static int threadsNumber;

public:
    Matrix(int rows, int columns);
    Matrix(const std::vector<std::vector<int>>& matrix);

    void fillMatrix();
    void printMatrix() const;

    int getRows() const;
    void setRows(int rows);
    int getColumns() const;
    void setColumns(int columns);

    static int getThreadsNumber();
    static void setThreadsNumber(int threadsNumber);

    int operator()(int row, int col) const;
    int& operator()(int row, int col);
    void setElement(int row, int column, int value);

    Matrix multiplyMatrix(const Matrix& other) const;
    Matrix multiplyMatrixByParallel(const Matrix& other) const;
};

#endif
