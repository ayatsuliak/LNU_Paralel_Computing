#pragma once
#ifndef MATRIX_H_
#define MATRIX_H_

#include <vector>
#include <complex>

using std::vector;
using std::complex;

template <typename T>
class Matrix
{
private:
	size_t rows, cols;
	vector<T> elements;
public:
	Matrix(size_t numOfRows, size_t numOfCols);
	Matrix(size_t numOfRows, size_t numOfCols, T* data);
	size_t getRows();
	size_t getCols();
	T operator()(size_t row, size_t col) const;
	T& operator()(size_t row, size_t col);
	T* data();
	const vector<T>& getElements();
};

#endif