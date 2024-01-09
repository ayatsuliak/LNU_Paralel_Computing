#include "dev_array.h"

template <class T>
void dev_array<T>::allocate(size_t size)
{
    cudaError_t result = cudaMalloc((void**)&start_, size * sizeof(T));
    if (result != cudaSuccess)
    {
        start_ = end_ = 0;
        throw std::runtime_error("failed to allocate device memory");
    }
    end_ = start_ + size;
}

template <class T>
void dev_array<T>::free()
{
    if (start_ != 0)
    {
        cudaFree(start_);
        start_ = end_ = 0;
    }
}

template <class T>
dev_array<T>::dev_array()
    : start_(0),
    end_(0)
{}

template <class T>
dev_array<T>::dev_array(size_t size)
{
    allocate(size);
}

template <class T>
dev_array<T>::~dev_array()
{
    free();
}

template <class T>
void dev_array<T>::resize(size_t size)
{
    free();
    allocate(size);
}

template <class T>
size_t dev_array<T>::getSize() const
{
    return end_ - start_;
}

template <class T>
const T* dev_array<T>::getData() const
{
    return start_;
}

template <class T>
T* dev_array<T>::getData()
{
    return start_;
}

template <class T>
void dev_array<T>::set(const T* src, size_t size)
{
    size_t min = std::min(size, getSize());
    cudaError_t result = cudaMemcpy(start_, src, min * sizeof(T), cudaMemcpyHostToDevice);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to device memory");
    }
}

template <class T>
void dev_array<T>::get(T* dest, size_t size)
{
    size_t min = std::min(size, getSize());
    cudaError_t result = cudaMemcpy(dest, start_, min * sizeof(T), cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        throw std::runtime_error("failed to copy to host memory");
    }
}