/*
* Some structures used in the kernels 
* Some modifications have been coded for a strategy pattern (accomodate more than one superpixel algorithm
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
*
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-0
* NLTVSaliencyCuda is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* NLTVSaliencyCuda is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with fastms. If not, see <http://www.gnu.org/licenses/>.
*
* Copyright 2020 Eduardo Alcain Ballesteros eduardo.alcain.ballesteros@gmail.com  Ana Muñoz anaisabel.munoz@urjc.es
*/
#ifndef __MYUTIL_H__
#define __MYUTIL_H__

#pragma once
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/extrema.h>
#include <thrust/random.h>
#include <thrust/sort.h>


// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type from https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf
template<class T>
struct SharedMemory
{
	__device__ inline operator T *()
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}

	__device__ inline operator const T *() const
	{
		extern __shared__ int __smem[];
		return (T *)__smem;
	}
};



// minmax_pair stores the minimum and maximum 
// values that have been encountered so far
template <typename T>
struct minmax_pair
{
	T min_val;
	T max_val;
};

// minmax_unary_op is a functor that takes in a value x and
// returns a minmax_pair whose minimum and maximum values
// are initialized to x.
template <typename T>
struct minmax_unary_op
	: public thrust::unary_function< T, minmax_pair<T> >
{
	__host__ __device__
		minmax_pair<T> operator()(const T& x) const
	{
		minmax_pair<T> result;
		result.min_val = x;
		result.max_val = x;
		return result;
	}
};

// minmax_binary_op is a functor that accepts two minmax_pair 
// structs and returns a new minmax_pair whose minimum and 
// maximum values are the min() and max() respectively of 
// the minimums and maximums of the input pairs
template <typename T>
struct minmax_binary_op
	: public thrust::binary_function< minmax_pair<T>, minmax_pair<T>, minmax_pair<T> >
{
	__host__ __device__
		minmax_pair<T> operator()(const minmax_pair<T>& x, const minmax_pair<T>& y) const
	{
		minmax_pair<T> result;
		result.min_val = thrust::min(x.min_val, y.min_val);
		result.max_val = thrust::max(x.max_val, y.max_val);
		return result;
	}
};


__device__ static float MyatomicMax(float* address, float val)
{
	int* address_as_i = (int*)address;
	int old = *address_as_i, assumed;
	do {
		assumed = old;
		old = ::atomicCAS(address_as_i, assumed,
			__float_as_int(::fmaxf(val, __int_as_float(assumed))));
	} while (assumed != old);
	return __int_as_float(old);
}


//Macro for checking cuda errors following a cuda launch or api call
#define cudaCheckError() {                                          \
 cudaError_t e=cudaGetLastError();                                 \
 if(e!=cudaSuccess) {                                              \
   printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(e));           \
   exit(0); \
 }                                                                 \
}

static const unsigned int nextPow2(unsigned int x)
{
	--x;
	x |= x >> 1;
	x |= x >> 2;
	x |= x >> 4;
	x |= x >> 8;
	x |= x >> 16;
	return ++x;
}


static bool isPow2(unsigned int x) {
	return ((x&(x - 1)) == 0);
}


#endif __MYUTIL_H__