// https://github.com/AJcodes/cuda_minmax/blob/master/cuda_minmax/kernel.cu

#ifndef __MIN_MAX_REDUCTION_H__
#define __MIN_MAX_REDUCTION_H__
#pragma once
#include <stdio.h>
#include <iostream>
#include "helper_cuda.h"
#include "../include/MyUtil.h"





template <class T, unsigned int blockSize>
__global__ void seq_minmaxKernel_(T *max, T *min, const T *a, int size) {
	__shared__ T maxtile[blockSize];
	__shared__ T mintile[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = a[i];
	mintile[tid] = a[i];
	__syncthreads();

	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
		if (tid < s && i + s < size) {
			//if (tid < s ) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}



template <class T, unsigned int blockSize>
__global__ void seq_maxKernel_(T *max, const T *a, int size) {
	__shared__ T maxtile[blockSize];
	

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = a[i];
	
	__syncthreads();

	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
		if (tid < s && i + s < size) {
			//if (tid < s ) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
		
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		
	}
}
template <class T, unsigned int blockSize>
__global__ void finalminmaxKernel_(T *max, T *min) {
	__shared__ T maxtile[blockSize];
	__shared__ T mintile[blockSize];

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];
	mintile[tid] = min[i];
	__syncthreads();

	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
		if (tid < s && tid + s < blockDim.x) {
			//if (tid < s ) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
			if (mintile[tid + s] < mintile[tid])
				mintile[tid] = mintile[tid + s];
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
		min[blockIdx.x] = mintile[0];
	}
}



template <class T, unsigned int blockSize>
__global__ void finalmaxKernel_(T *max) {
	__shared__ T maxtile[blockSize];
	

	unsigned int tid = threadIdx.x;
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	maxtile[tid] = max[i];

	__syncthreads();

	//sequential addressing by reverse loop and thread-id based indexing
	for (unsigned int s = blockSize / 2; s > 0; s >>= 1) {
		if (tid < s && tid + s < blockDim.x) {
			//if (tid < s ) {
			if (maxtile[tid + s] > maxtile[tid])
				maxtile[tid] = maxtile[tid + s];
		
		}
		__syncthreads();
	}

	if (tid == 0) {
		max[blockIdx.x] = maxtile[0];
	
	}
}



////////////////////////////////////////////////////////////////////////////////
// Wrapper function for kernel launch
////////////////////////////////////////////////////////////////////////////////
template <class T>
void
minmax(int size, T *d_max, T *d_min, const T *d_vector)
{
	//std::cout << "MIN MAX" << std::endl;
	int blocksize = 128;

	int grid = (size / blocksize) + ((size % blocksize == 0) ? 0 : 1);
	dim3 dimBlock(blocksize);
	dim3 dimGrid(grid);
	//mykernel<float><< < 1, 1>>>(d_min);
	seq_minmaxKernel_<T, 128> << <dimGrid, dimBlock >> >(d_max, d_min, d_vector, size);
	switch (nextPow2(grid)) {
	case 2:
		finalminmaxKernel_<T, 2> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 4:
		finalminmaxKernel_<T, 4> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 8:
		finalminmaxKernel_<T, 8> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 16:
		finalminmaxKernel_<T, 16> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 32:
		finalminmaxKernel_<T, 32> << <1, dimGrid >> > (d_max, d_min);
	case 64:
		finalminmaxKernel_<T, 64> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 128:
		finalminmaxKernel_<T, 128> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 256:
		finalminmaxKernel_<T, 256> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 512:
		finalminmaxKernel_<T, 512> << <1, dimGrid >> > (d_max, d_min);
		break;
	case 1024:
		finalminmaxKernel_<T, 1024> << <1, dimGrid >> > (d_max, d_min);
		break;
	}
}
// To Review the blocksize for the min max and max elements
// have a look at mixMaxCuda project for atomic operation to speed up the process.
template <class T>
T maxElements(int size, const T *d_vector)
{
	//std::cout << "MIN MAX" << std::endl;
	int blocksize = 1024;
	T * d_max;
	int grid = (size / blocksize) + ((size % blocksize == 0) ? 0 : 1);
	dim3 dimBlock(blocksize);
	dim3 dimGrid(grid);

	
	checkCudaErrors(cudaMalloc((void**)&d_max, grid * sizeof(float)));
	//mykernel<float><< < 1, 1>>>(d_min);
	seq_maxKernel_<T, 1024> << <dimGrid, dimBlock >> >(d_max, d_vector, size);
	switch (nextPow2(grid)) {
	case 2:
		finalmaxKernel_<T, 2> << <1, dimGrid >> > (d_max);
		break;
	case 4:
		finalmaxKernel_<T, 4> << <1, dimGrid >> > (d_max);
		break;
	case 16:
		finalmaxKernel_<T, 16> << <1, dimGrid >> > (d_max);
		break;
	case 32:
		finalmaxKernel_<T, 32> << <1, dimGrid >> > (d_max);
	case 64:
		finalmaxKernel_<T, 64> << <1, dimGrid >> > (d_max);
		break;
	case 128:
		finalmaxKernel_<T, 128> << <1, dimGrid >> > (d_max);
		break;
	case 256:
		finalmaxKernel_<T, 256> << <1, dimGrid >> > (d_max);
		break;
	case 512:
		finalmaxKernel_<T, 512> << <1, dimGrid >> > (d_max);
		break;
	case 1024:
		finalmaxKernel_<T, 1024> << <1, dimGrid >> > (d_max);
		break;
	default:
		std::cout << "Max Elements no possible final kernel to reduce" << grid << std::endl;
		break;
	}
	
	T max;
	checkCudaErrors(cudaMemcpy(&max, d_max, sizeof(T), cudaMemcpyDeviceToHost));
	cudaFree(d_max);
	return max;
}

// Instantiate the reduction function for 3 types
template void
minmax<int>(int size, int *d_max, int *d_min, const int *d_vector);

template void
minmax<float>(int size, float *d_max, float *d_min, const  float *d_vector);

template void
minmax<double>(int size, double *d_max, double *d_min, const double *d_vector);




template int
maxElements<int>(int size,  const int *d_vector);

template float
maxElements<float>(int size,  const  float *d_vector);

template double
maxElements<double>(int size,  const double *d_vector);


#endif // #ifndef __MIN_MAX_REDUCTION_H__

