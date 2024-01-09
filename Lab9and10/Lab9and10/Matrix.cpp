#include "matrix.h"

template <typename T>
Matrix<T>::Matrix(size_t numOfRows, size_t numOfCols)
	: rows(numOfRows), cols(numOfCols), elements(numOfRows* numOfCols) {}

template <typename T>
Matrix<T>::Matrix(size_t numOfRows, size_t numOfCols, T* data)
	: rows(numOfRows), cols(numOfCols), elements(data, data + numOfRows * numOfCols) {}

template <typename T>
size_t Matrix<T>::getRows()
{
	return rows;
}

template <typename T>
size_t Matrix<T>::getCols()
{
	return cols;
}

template <typename T>
T Matrix<T>::operator()(size_t row, size_t col) const
{
	return elements[cols * row + col];
}

template <typename T>
T& Matrix<T>::operator()(size_t row, size_t col)
{
	return elements[cols * row + col];
}

template <typename T>
T* Matrix<T>::data()
{
	return elements.data();
}

template <typename T>
const vector<T>& Matrix<T>::getElements()
{
	return elements;
}