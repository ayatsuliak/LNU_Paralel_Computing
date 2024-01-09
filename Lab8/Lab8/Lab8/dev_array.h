#pragma once
#ifndef _DEV_ARRAY_H_
#define _DEV_ARRAY_H_

#include <stdexcept>
#include <algorithm>
#include <cuda_runtime.h>

template <class T>
class dev_array
{
private:
	T* start_;
	T* end_;

	void allocate(size_t size);
	void free();
public:
	explicit dev_array();
	explicit dev_array(size_t size);
	~dev_array();
	void resize(size_t size);
	size_t getSize() const;
	const T* getData() const;
	T* getData();
	void set(const T* src, size_t size);
	void get(T* dest, size_t size);
};

#endif