#include "matrix.h"

int Matrix::threadsNumber = 1;

Matrix::Matrix(int rows, int columns) : rows(rows), columns(columns), matrix(rows, std::vector<int>(columns, 0)) {}

Matrix::Matrix(const std::vector<std::vector<int>>& matrix) : matrix(matrix) {
    if (!matrix.empty()) {
        rows = matrix.size();
        columns = matrix[0].size();
    }
    else {
        rows = columns = 0;
    }
}

void Matrix::fillMatrix() {
    std::srand(static_cast<unsigned int>(std::time(nullptr)));
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            matrix[i][j] = std::rand() % 101;
        }
    }
}

void Matrix::printMatrix() const {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            std::cout << matrix[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int Matrix::getRows() const {
    return rows;
}

void Matrix::setRows(int rows) {
    this->rows = rows;
    matrix.resize(rows);
}

int Matrix::getColumns() const {
    return columns;
}

void Matrix::setColumns(int columns) {
    this->columns = columns;
    for (int i = 0; i < rows; ++i) {
        matrix[i].resize(columns);
    }
}

int Matrix::getThreadsNumber() {
    return threadsNumber;
}

void Matrix::setThreadsNumber(int threadsNumber) {
    Matrix::threadsNumber = threadsNumber;
}

int Matrix::operator()(int row, int col) const
{
    return matrix[row][col];
}

int& Matrix::operator()(int row, int col)
{
    return matrix[row][col];
}

void Matrix::setElement(int row, int column, int value) {
    matrix[row][column] = value;
}

Matrix Matrix::multiplyMatrix(const Matrix& other) const {
    if (columns != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    int resultRows = rows;
    int resultColumns = other.columns;
    Matrix resultMatrix(resultRows, resultColumns);

    for (int i = 0; i < resultRows; ++i) {
        for (int j = 0; j < resultColumns; ++j) {
            int sum = 0;
            for (int k = 0; k < columns; ++k) {
                sum += (i, k) * other(k, j);
            }
            resultMatrix.setElement(i, j, sum);
        }
    }

    return resultMatrix;
}

Matrix Matrix::multiplyMatrixByParallel(const Matrix& other) const {
    if (columns != other.rows) {
        throw std::invalid_argument("Matrix dimensions do not match for multiplication.");
    }

    int resultRows = rows;
    int resultColumns = other.columns;
    Matrix resultMatrix(resultRows, resultColumns);
    std::vector<std::thread> threads;
    std::mutex mtx;

    for (int i = 0; i < resultRows; ++i) {
        threads.emplace_back([this, &other, &resultMatrix, i, &mtx, resultColumns] {
            for (int j = 0; j < resultColumns; ++j) {
                int sum = 0;
                for (int k = 0; k < columns; ++k) {
                    sum += matrix[i][k] * other(k, j);
                }
                mtx.lock();
                resultMatrix.setElement(i, j, sum);
                mtx.unlock();
            }
            });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    return resultMatrix;
}
