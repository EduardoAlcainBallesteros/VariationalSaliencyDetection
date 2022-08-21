
/*
* Kernels use for the implementation of the following saliency models
* E(u) = J_{NLTV,w}(u) + \lambda F(u)				(Saliency NLTV )
* E(u) = J_{NLTV,w}(u) + \lambda F(u) - H(u)		(Saleincy NLTV with SaliencyTerm)
* - J_{NLTV,w} ({\mathbf u})=\sum_{p \in V} \left( \sum_{q \in V,\, pq\in E} w_{pq}|u_q-u_p|^2 \right)^{1/2}
* - F(\mathbf{u})=\frac{1}{\alpha}  ||\mathbf{u}-\mathbf{v}^c ||^2 = \frac{1}{2\alpha} \sum_{p\in V}  |u_p-v^c_p|^2
* - H(\mathbf{u})=\frac{1}{2\alpha^2}\sum_{p\in V} (1-\delta u_p)^2,
* - \lamdbda The parameter is a positive constant for controlling  the relative importance of the regularizer vs the delity in the functional
* Reductions have been implemented by https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf. Better performance could be achieved by https://nvlabs.github.io/cub/
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-01016-4
* * NLTVSaliencyCuda is free software: you can redistribute it and/or modify
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
* Copyright 2020 Eduardo Alcain Ballesteros eduardo.alcain.ballesteros@gmail.com Ana Muñoz anaisabel.munoz@urjc.es
*/
#ifndef __NLTV_Saliency_Alg_Kernel_CU__
#define __NLTV_Saliency_Alg_Kernel_CU__
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../include/MyUtil.h"
#include "../include/MinMax.h"
#include "../include/MatrixIO.h"
#include "helper_cuda.h"
#include <cublas_v2.h>
#include <stdio.h>
#ifdef DEBUG_MATRIX	 
#include "MatrixMatFile.h"
#endif

// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
// https://github.com/thrust/thrust/blob/master/examples/minmax.cu


// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
template <typename T>
__global__ void RgbToLabKernel(const int* d_rin, const int* d_gin, const int* d_bin, const int sz, T* d_L, T* d_a, T* d_b) {
	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < sz;
		i += blockDim.x * gridDim.x)
	{

		int sR, sG, sB;
		T R, G, B;
		T X, Y, Z;
		T r, g, b;
		const T epsilon = 0.008856f;	//actual CIE standard
		const T kappa = 903.3f;		//actual CIE standard

		const T Xr = 0.950456f;	//reference white
		const T Yr = 1.0f;		//reference white
		const T Zr = 1.088754f;	//reference white
		T xr, yr, zr;
		T fx, fy, fz;
		T lval, aval, bval;


		sR = d_rin[i]; sG = d_gin[i]; sB = d_bin[i];
		R = sR / 255.0f;
		G = sG / 255.0f;
		B = sB / 255.0f;

		if (R <= 0.04045f)	r = R / 12.92;
		else				r = pow((R + 0.055) / 1.055, 2.4);
		if (G <= 0.04045f)	g = G / 12.92;
		else				g = pow((G + 0.055) / 1.055, 2.4);
		if (B <= 0.04045f)	b = B / 12.92;
		else				b = pow((B + 0.055) / 1.055, 2.4);

		X = r*0.4124564f + g*0.3575761f + b*0.1804375f;
		Y = r*0.2126729f + g*0.7151522f + b*0.0721750f;
		Z = r*0.0193339f + g*0.1191920f + b*0.9503041f;

		//------------------------
		// XYZ to LAB conversion
		//------------------------
		xr = X / Xr;
		yr = Y / Yr;
		zr = Z / Zr;

		if (xr > epsilon)	fx = pow(xr, 1.0f / 3.0f);
		else				fx = (kappa*xr + 16.0f) / 116.0f;
		if (yr > epsilon)	fy = pow(yr, 1.0f / 3.0f);
		else				fy = (kappa*yr + 16.0f) / 116.0f;
		if (zr > epsilon)	fz = pow(zr, 1.0f / 3.0f);
		else				fz = (kappa*zr + 16.0f) / 116.0f;

		lval = 116.0f*fy - 16.0f;
		aval = 500.0f*(fx - fy);
		bval = 200.0f*(fy - fz);

		d_L[i] = lval; d_a[i] = aval; d_b[i] = bval;
	}
}

template <typename T>
__global__ void CreateSuperpixelAttributeKernel(int n,  const int *d_labels, const T *d_l, const T* d_a, const T* d_b, const int outputNumSuperpixels, T *d_lSp, T  *d_aSp, T *d_bSp, T *d_rSp, T *d_cSp, int *d_numSp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		atomicAdd(&d_lSp[d_labels[i]], d_l[i]);
		atomicAdd(&d_aSp[d_labels[i]], d_a[i]);
		atomicAdd(&d_bSp[d_labels[i]], d_b[i]);
		atomicAdd(&d_rSp[d_labels[i]], blockIdx.x);
		atomicAdd(&d_cSp[d_labels[i]], threadIdx.x);
		atomicAdd(&d_numSp[d_labels[i]], 1);
	}
}

template <typename T>
__global__ void DivideNumSuperpixelKernel(int n, T *d_L, T* d_a, T* d_b, T *d_r, T *d_c,const int *numSp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	// numSp[i]>0 gSLICr gives you sp with no value in the region
	if (i < n && numSp[i]>0) {		
		d_L[i] = d_L[i] / (T)numSp[i];
		d_a[i] = d_a[i] / (T)numSp[i];
		d_b[i] = d_b[i] / (T)numSp[i];
		d_r[i] = d_r[i] / (T)numSp[i];
		d_c[i] = d_c[i] / (T)numSp[i];
	}
}

template <typename T>
__global__ void NormalizeDataKernel(int n, T *d_L, T* d_a, T* d_b, T *d_r, T *d_c, const T *maxValueL_, const T *maxValuea_, const T *maxValueb_, const T * maxValuer_, const T *maxValuec_,
	const T *minValueL_, const T *minValuea_, const T *minValueb_, const T *minValuer_, const T *minValuec_) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		
		d_L[i] = (d_L[i] - *minValueL_) / (*maxValueL_ - *minValueL_);
		d_a[i] = (d_a[i] - *minValuea_) / (*maxValuea_ - *minValuea_);
		d_b[i] = (d_b[i] - *minValueb_) / (*maxValueb_ - *minValueb_);
		d_r[i] = (d_r[i] - *minValuer_) / (*maxValuer_ - *minValuer_);
		d_c[i] = (d_c[i] - *minValuec_) / (*maxValuec_ - *minValuec_);

	}
}


template <typename T,bool nIsPow2>
__global__ void CalculateSaliencyControlMapAndWeightsKernel(T *d_saliencyControlMap, const  T *d_a, const  T *d_b, const  T *d_L, const  T *d_r, const  T *d_c, int numSuperpixels, bool locationPrior,  T *d_weights, const T sigma2,const T r) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	T v_p_obj = 0, w_pq = 0;
	
	T sum;
	T l_i_r, l_i_c, l_j_r, l_j_c, a_i, b_i, L_i;
	T aux;
	// Normalize
	l_i_r = d_r[i];
	l_i_c = d_c[i];
	a_i = d_a[i];
	b_i = d_b[i];
	L_i = d_L[i];
	T  l_bar_r, l_bar_c;
	l_bar_r = 0.5f;
	l_bar_c = 0.5f;

	/** Contrast prior PART */
	
	// If i = j the result is 0 
	// Normalize
	l_j_r = d_r[j];
	l_j_c = d_c[j];
	// Calculate
	aux = (l_i_r - l_j_r) * (l_i_r - l_j_r) +
		(l_i_c - l_j_c) * (l_i_c - l_j_c);
	w_pq = exp((-1 * aux) / (sigma2));

	aux = (a_i - d_a[j]) * (a_i - d_a[j]) +
		(b_i - d_b[j]) * (b_i - d_b[j]) +
		(L_i - d_L[j]) * (L_i - d_L[j]);
	// Formula (4) Write in th SM the result for the sum later
	sum = i != j ? w_pq* aux :0.0;
	//sum = i != j ? 1: 0;
	sum = blockReduceSum(sum, nIsPow2);
	/** Contrast prior PART */

	/** WEIGHTS PART */

	

	T aAux = r*d_a[i];
	T bAux = r*d_b[i];
	T LAux = r*d_L[i];
	T rAux = d_r[i];
	T cAux = d_c[i];


	T aAuxII = r* d_a[j];
	T bAuxII = r* d_b[j];
	T LAuxII = r* d_L[j];
	T exponentValue = (aAux - aAuxII) * (aAux - aAuxII) +
		(bAux - bAuxII) * (bAux - bAuxII) +
		(LAux - LAuxII) * (LAux - LAuxII) +
		(rAux - d_r[j]) * (rAux - d_r[j]) +
		(cAux - d_c[j]) * (cAux - d_c[j]);

	aux = exp(-1 * exponentValue / (sigma2));
	if (blockIdx.x != threadIdx.x) {
		d_weights[index] = aux;
	}
	else {
		d_weights[index] = 0.0;
	}

	/** WEIGHTS PART END  */



	/** Contrast prior PART Reduction ends*/

	/** Calculation Control Map for j superpixel */

	if (j == 0) {
		aux = (l_i_r - l_bar_r) * (l_i_r - l_bar_r) +
			(l_i_c - l_bar_c) * (l_i_c - l_bar_c);
		if (locationPrior) {
			// Object Prior calcultaion Formula (5)
			v_p_obj = exp((-1 * aux) / (sigma2));
		}
		else {

			v_p_obj = 1;
		}
		// Saliency control map Formula (6)
		
		d_saliencyControlMap[i] = sum * v_p_obj;
		//d_saliencyControlMap[i] = sum ;
		
	}
	/** Calculation Control Map for j superpixel ends */

}

template <typename T>
__global__ void NormalizeKernel(T *d_saliencyControlMap, T range, T minValue, const int numSuperixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperixels) {
		d_saliencyControlMap[i] = (d_saliencyControlMap[i] - minValue) / range;
	}
}

template <typename T>
__global__ void NormalizeKernel_(T *d_saliencyControlMap, const  T *minValue, const T *maxValue, const int numSuperixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperixels) {
		T range = *maxValue - *minValue;
		d_saliencyControlMap[i] = (d_saliencyControlMap[i] - *minValue) / range;
	}
}

template <typename T, unsigned int blockSize>
__global__ void CalculateKNNKernel(T* d_weights,const int knn, const int width) {

	bool setZero = true;
	int j = threadIdx.x;
	int indexMaxIter;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	T threadValue;
	T *sdata = SharedMemory<T>();
	__shared__ int maxs[blockSize];
	threadValue = d_weights[index];
	sdata[j] = threadValue;
	maxs[j] = j;

	__syncthreads();

	// do reduction in shared mem
	for (int h = 0; h < knn; h++) {
	
	
		for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
		{
			if (j < s && j + s < blockDim.x)
			{

				if (sdata[j + s] > sdata[j]) {
					sdata[j] = sdata[j + s];
					maxs[j] = maxs[j + s];
				}
			}
			__syncthreads();
		}
		indexMaxIter = maxs[0];
		setZero = setZero && !(indexMaxIter == j);
		if (!setZero) {
			threadValue = 0;
			sdata[j] = threadValue;
			maxs[j] = j;
		}
		else {
			sdata[j] = threadValue;
			maxs[j] = j;
		}
		__syncthreads();
	
		
		
	}
	if(setZero)
		d_weights[index] = 0.0;
	
}

// Old 
template <typename T, unsigned int blockSize>
__global__ void CalculateKNNKernelOld(T* d_weights,const int knn, const int width) {

	bool setZero = true;
	int j = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	T *sdata = SharedMemory<T>();
	__shared__ int maxsFinal[5];
	__shared__ int maxs[blockSize];


	// do reduction in shared mem
	for (int h = 0; h < knn; h++) {
		sdata[j] = d_weights[index];
		maxs[j] = j;

		__syncthreads();
		for (int k = 0; k < h; k++) {
			sdata[maxsFinal[k]] = 0;
		}
		__syncthreads();
		for (unsigned int s = blockSize / 2; s > 0; s >>= 1)
		{
			if (j < s && j + s < blockDim.x)
			{

				if (sdata[j + s] > sdata[j]) {
					sdata[j] = sdata[j + s];
					maxs[j] = maxs[j + s];
				}
			}
			__syncthreads();
		}
		maxsFinal[h] = maxs[0];
		setZero = setZero && !(maxsFinal[h] == j);
		__syncthreads();
	}
	if (setZero)
		d_weights[index] = 0.0;
}

template <typename T>
__global__ void CalculateBKernel(const  T *d_saliencyControlMap, T *d_bPD, const T  deltaSalTerm, const T alphaSalTerm, const T lambda, const T a, int numSuperpixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixels) {
		d_bPD[i] = (deltaSalTerm / (alphaSalTerm * alphaSalTerm)) - (lambda / alphaSalTerm)*d_saliencyControlMap[i];
	}
}
//// Iterative 
template <typename T>
__global__ void CalculateTransIndexKernel(int *d_trans, const int  *d_rows, const int * d_cols, const T * d_values, const int knn, const int N) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < N) {
		int aux = p* knn;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int row = d_cols[jj];
			int col = p;
			int val = -1;
			for (int h = d_rows[row], z = 0; d_cols[h] <= col && h < d_rows[row + 1]; h++, z++) {
				if (d_cols[h] == col) {
					val = z;
					break;
				}
			}
			d_trans[aux + k] = val;
		}
	}
}

template <typename T>
__global__ void NLTVGradientKernel(const int  *d_rows, const int * d_cols, const T * d_values, const T * uk, T * d_pk, const int k, const T tau_d, int _N) {
	int i = blockIdx.x * blockDim.x;
	int j = threadIdx.x;
	int index = i + j;
	T maxValue = 0;
	//d_pk[index] = index;
	if (index < _N) {
		//int index = i / k;
		int col = index - i;
		//int col = i % k;
		int row_start = d_rows[blockIdx.x];
		int row_end = d_rows[blockIdx.x + 1];
		//T aux = d_pk[i] / max[iter];
		T aux = d_pk[index];
		if ((row_end - row_start) > col) {
			int jj = row_start + col;

			d_pk[index] = aux + tau_d * (uk[d_cols[jj]] - uk[blockIdx.x])* sqrt(d_values[jj]);
		
		}
	


	}
}

template <typename T, bool nIsPow2>
__global__ void Max(T * d_max, const  T * d_pk, const int _N) {
	int i = blockIdx.x * blockDim.x;
	int j = threadIdx.x;
	int index = i + j;
	T maxValue = 0;
	//d_pk[index] = index;
	if (index < _N) {
		
		maxValue = d_pk[index];
		
		maxValue = maxValue*maxValue;
		maxValue = blockReduceSum(maxValue, false);

		float aux = sqrt((float)maxValue);
		if (threadIdx.x ==0 && maxValue>1) {
		
			d_max[blockIdx.x] = maxValue;
		}
		else {
			d_max[blockIdx.x] = 1.0;
		}


	}
}


template <typename T>
__global__ void DivByMaxKernel(const T * d_max, const int N, T *d_pk) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		d_pk[i] = d_pk[i] / d_max[blockIdx.x];
	}
}

__inline__ __device__
float warpReduceSum(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {
		val += __shfl_down(val, offset);
	}
	return val;
}

__inline__ __device__
float warpAllReduceSum(float val) {
	for (int mask = warpSize / 2; mask > 0; mask /= 2)
		val += __shfl_xor(val, mask);
	return val;
}

__inline__ __device__
float blockReduceSum(float val, bool nIsPow2) {

	static __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceSum(val);     // Each warp performs partial reduction
								  //val = warpAllReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

								  //read from shared memory only if that warp existed
	if (nIsPow2)
		val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
	else
		val = (threadIdx.x <= blockDim.x / warpSize) ? shared[lane] : 0.0f;

	if (wid == 0) val = warpReduceSum(val); //Final reduce within first warp
	

	return val;
}

__inline__ __device__
float warpReduceMax(float val) {
	for (int offset = warpSize / 2; offset > 0; offset /= 2) {

		float aux = __shfl_down(val, offset);
		if (val < aux)
			val = aux;
	}

	return val;
}

__inline__ __device__
float blockReduceMax(float val, bool nIsPow2) {

	static __shared__ float shared[32]; // Shared mem for 32 partial sums
	int lane = threadIdx.x % warpSize;
	int wid = threadIdx.x / warpSize;

	val = warpReduceMax(val);     // Each warp performs partial reduction
								  //val = warpAllReduceSum(val);     // Each warp performs partial reduction

	if (lane == 0) shared[wid] = val; // Write reduced value to shared memory

	__syncthreads();              // Wait for all partial reductions

								  //read from shared memory only if that warp existed
								  //val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;
	if (nIsPow2)
		val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0.0f;
	else
		val = (threadIdx.x <= blockDim.x / warpSize) ? shared[lane] : 0.0f;

	if (wid == 0) val = warpReduceMax(val); //Final reduce within first warp

	return val;
}


template <typename T, bool nIsPow2>
__global__ void MaximizationNLTVSaltermKernel(const int  *d_rows, const int * d_cols, const T * d_values, const T * uk, T * d_pk, const int k, const T tau_d, int _N) {
	int i = blockIdx.x * blockDim.x ;
	int j = threadIdx.x;
	int index = i+j;
	T maxValue = 0;
	//d_pk[index] = index;
	if (index < _N) {
		//int index = i / k;
		int col = index - i;
		//int col = i % k;
		int row_start = d_rows[blockIdx.x];
		int row_end = d_rows[blockIdx.x + 1];
		//T aux = d_pk[i] / max[iter];
		T aux = d_pk[index];
		if ((row_end - row_start) > col) {


			int jj = row_start + col;

			d_pk[index] = aux + tau_d * (uk[d_cols[jj]] - uk[blockIdx.x])* sqrt(d_values[jj]);
			aux = d_pk[index];
		}
		maxValue = aux*aux;
		maxValue = blockReduceSum(maxValue, false);
		
		d_pk[index] = sqrt((float)maxValue);
		__syncthreads();
		if (d_pk[blockIdx.x * blockDim.x]>1) {
			d_pk[index] = aux / d_pk[blockIdx.x * blockDim.x];
		}
		else {
			d_pk[index] = aux ;
		}
		
		
	
	}
}

template <typename T>
__global__ void NLTVDivKernel(const  T *d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, const T * uk, const T * pMatrix, T *divP, const int knn, const T tau_d, const int N) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//int p = i / knn;
	if (p < N) {
		int aux = p* knn;
		T sumValue = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int row = d_cols[jj];
			int col = p;
			float val = 0.0;
			for (int h = d_rows[row], z = 0; d_cols[h] <= col && h < d_rows[row + 1]; h++, z++) {
				if (d_cols[h] == col) {
					val = pMatrix[row*knn + z];
					break;
				}
			}
			sumValue = sumValue + (pMatrix[aux + k] - val)* sqrt(d_values[jj]);
		}
		divP[p] = sumValue;
	}
}

template <typename T, bool nIsPow2>
__global__ void MinimizationNLTVSalTermEnergyKernel(const  T *d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, const int * d_trans, const T * pMatrix, const int knn, const T tau_d, const int N, T * uk, const T * b, const T tau_p, const T a, T *term, const T delta) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	T sum;
	if (p < N) {

		T sumValue = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {

			int index = p*knn + k;
			T val = 0.0;
			if (d_trans[index] != -1) {
				int row = d_cols[jj];
				int col = d_trans[index];
				//val = (pMatrix[row*knn + col]) / max[iter];
				val = (pMatrix[row*knn + col]);
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			//sumValue = sumValue + ((pMatrix[p* knn + k] - val) / max[iter])* sqrt(d_values[jj]);
			sumValue = sumValue + ((pMatrix[p* knn + k] - val) )* sqrt(d_values[jj]);
		}
		//T divP = sumValue;
		sumValue = (1 + a * tau_p) * uk[p] + tau_p *(sumValue - b[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;

		sum = (1 - delta*sumValue)*(1 - delta*sumValue);
		sum = blockReduceSum(sum, nIsPow2);

		if (p == 0) term[2] = sum;

		// (uk[p] - d_saliencyControlMap[p])*(uk[p] - d_saliencyControlMap[p]);
		sum = (sumValue - d_saliencyControlMap[p])*(sumValue - d_saliencyControlMap[p]);
		sum = blockReduceSum(sum, nIsPow2);
		if (p == 0) term[1] = sum;

		uk[p] = sumValue;

		__syncthreads();
		sumValue = 0.0;
		row_start = d_rows[p];
		row_end = d_rows[p + 1];
		for (int jj = row_start; jj < row_end; jj++) {

			int q = d_cols[jj];
			sumValue = sumValue + ((uk[q] - uk[p]) * (uk[q] - uk[p]))* d_values[jj];
			
		}
		sumValue = sqrt(sumValue);
		
		sumValue = blockReduceSum(sumValue, nIsPow2);
		if (p == 0) term[0] = sumValue;
	}
}

template <typename T>
__global__ void MinimizationNLTVSalTermNoEnergyKernel(const  T *d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, const int * d_trans, const T * pMatrix, const int knn, const T tau_d, const int N, T * uk, const T * b, const T tau_p, const T a) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < N) {

		T sumValue = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int index = p*knn + k;
			T val = 0.0;
			if (d_trans[index] != -1) {
				int row = d_cols[jj];
				int col = d_trans[index];
				val = (pMatrix[row*knn + col]) ;
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ((pMatrix[p* knn + k] - val))* sqrt(d_values[jj]);
		}

		sumValue = (1 + a * tau_p) * uk[p] + tau_p *(sumValue - b[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;
		uk[p] = sumValue;
	}
}

template <typename T, bool nIsPow2 >
__global__ void MinimizationNLTVEnergyKernel(const  T *d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, const int * d_trans, const T * pMatrix, const int knn, const T tau_d, const int N, T * uk, const T tau_p, const T divA, T *term) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	T sum;

	if (p < N) {

		T sumValue = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int index = p*knn + k;
			T val = 0.0;
			if (d_trans[index] != -1) {
				int row = d_cols[jj];
				int col = d_trans[index];
				val = (pMatrix[row*knn + col]) ;
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ((pMatrix[p* knn + k])  - val)* sqrt(d_values[jj]);
		}
		sumValue = (1 - tau_p) * uk[p] + tau_p *(divA *sumValue + d_saliencyControlMap[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;

		// (uk[p] - d_saliencyControlMap[p])*(uk[p] - d_saliencyControlMap[p]);
		sum = (sumValue - d_saliencyControlMap[p])*(sumValue - d_saliencyControlMap[p]);
		sum = blockReduceSum(sum, nIsPow2);

		if (p == 0) term[1] = sum;

		uk[p] = sumValue;
		__syncthreads();
		sumValue = 0.0;
		row_start = d_rows[p];
		row_end = d_rows[p + 1];
		for (int jj = row_start; jj < row_end; jj++) {

			int q = d_cols[jj];
			sumValue = sumValue + (uk[q] - uk[p]) * (uk[q] - uk[p])* d_values[jj];
			
		}
		sumValue = sqrt(sumValue);
		sumValue = blockReduceSum(sumValue, nIsPow2);
		if (p == 0) term[0] = sumValue;
	}
}

template <typename T>
__global__ void MinimizationNLTVNoEnergyKernel(const  T *d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, const int * d_trans, const T * pMatrix, const int knn, const T tau_d, const int N, T * uk, const T tau_p, const T divA) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;

	if (p < N) {

		T sumValue = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int index = p*knn + k;
			T val = 0.0;
			if (d_trans[index] != -1) {
				int row = d_cols[jj];
				int col = d_trans[index];
				val = (pMatrix[row*knn + col]) ;
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ((pMatrix[p* knn + k]) - val)* sqrt(d_values[jj]);
		}
		//T divP = sumValue;
		sumValue = (1 - tau_p) * uk[p] + tau_p *(divA *sumValue + d_saliencyControlMap[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;
		uk[p] = sumValue;
	}
}

template <typename T>
__global__ void UpdateVariableNLTVSalTermKernel(const  T *d_saliencyControlMap, T * uk, const T * b, const T *divP, const T tau_p, const T a, const int numSuperpixel, T * d_term) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixel) {

		T aux = (1 + a *tau_p) * uk[i] + tau_p *(divP[i] - b[i]);
		if (aux < 0.0)
			aux = 0.0;
		else if (aux > 1.0)
			aux = 1.0;
		uk[i] = aux;
		if (i < 3) {
			d_term[i] = 0.0;
		}
	}
}

template <typename T>
__global__ void UpdateVariableNLTVKernel(const  T *d_saliencyControlMap, T * uk, const T *divP, const T tau_p, const T divLambda, const int numSuperpixel, T * d_term) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixel) {
		//uk[i] = (1 - tau_p) * uk[i] + tau_p *(divA *divP[i] + b[i]);
		T aux = (1 - tau_p) * uk[i] + tau_p *(divLambda* divP[i] + d_saliencyControlMap[i]);
		if (aux < 0.0)
			aux = 0.0;
		else if (aux > 1.0)
			aux = 1.0;
		uk[i] = aux;
		if (i < 3) {
			d_term[i] = 0.0;
		}
	}
}

template <typename T>
__global__ void EnergyNLTVSalTermKernel(const int _N, const T * uk, const T * d_saliencyControlMap, const T delta, T *term) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < _N) {
		T term1Aux = 0.0;
		for (int q = 0; q < _N; q++) {

			term1Aux = term1Aux + ((uk[q] - uk[p]) * (uk[q] - uk[p]));
		}

		atomicAdd(&term[0], sqrt(term1Aux));
		atomicAdd(&term[1], (uk[p] - d_saliencyControlMap[p])*(uk[p] - d_saliencyControlMap[p]));
		atomicAdd(&term[2], (1 - delta* uk[p])*(1 - delta* uk[p]));
		
	}
}

template <typename T>
__global__ void EnergyNLTVKernel(const int _N, const T * uk, const T * d_saliencyControlMap, const int  *d_rows, const int * d_cols, const T * d_values, T lambda, T *term) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < _N) {
		T term1Aux = 0.0;
		int row_start = d_rows[p];
		int row_end = d_rows[p + 1];
		/*for (int q = 0; q < _N; q++) {

			term1Aux = term1Aux + ((uk[q] - uk[p]) * (uk[q] - uk[p]));
		}*/
		for (int index = row_start; index < row_end; index++) {

			term1Aux = term1Aux + (d_values[index] * (uk[d_cols[index]] - uk[p]) * (uk[d_cols[index]] - uk[p]));
			
			//term1Aux = term1Aux + ((uk[d_cols[index]] - uk[p]) * (uk[d_cols[index]] - uk[p]));
		}

		
		atomicAdd(&term[0], sqrt(term1Aux));
		atomicAdd(&term[1], (uk[p] - d_saliencyControlMap[p])*(uk[p] - d_saliencyControlMap[p]));
	}
}

// ENDS ITERATIVE KERNELS

template <typename T>
void RgbToLabGpu(const int* d_rin, const int* d_gin,const int* d_bin,const int sz, T* d_L, T* d_a, T* d_b) {
	int numThreads = 256;
	int numBlocks = (sz / numThreads);
	int extra = (sz % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	RgbToLabKernel << <numBlocks, numThreads >> > (d_rin, d_gin, d_bin, sz, d_L, d_a, d_b);
	

	//return cudaGetLastError();
}
//
template <typename T>
void CreateSuperpixelAttributeGpu(int height, int width, const int *d_labels, const T *d_l, const T* d_a, const T* d_b, const int sp, T *d_lSp, T  *d_aSp, T *d_bSp, T *d_rSp, T *d_cSp, int *numSp) {
	int n = height *width;
	T *d_max_a = 0;
	T *d_min_a = 0;
	T *d_min_b = 0;
	T *d_max_b = 0;
	T *d_min_L = 0;
	T *d_max_L = 0;
	T *d_min_r = 0;
	T *d_max_r = 0;
	T *d_min_c = 0;
	T *d_max_c = 0;

	
	// Launch a kernel on the GPU with one thread for each element.
	int numThreads = 128;
	int numBlocks = (sp / numThreads);
	int extra = (sp % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start);
	cudaError_t cudaStatus;
	
	CreateSuperpixelAttributeKernel << < height, width >> > (n, d_labels, d_l, d_a, d_b, sp, d_lSp, d_aSp, d_bSp, d_rSp, d_cSp, numSp);
	
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");

	}
		
	DivideNumSuperpixelKernel << <1, sp >> >(sp, d_lSp, d_aSp, d_bSp, d_rSp, d_cSp, numSp);
	int blocksize = 128;
	int grid = (sp / blocksize) + ((sp % blocksize == 0) ? 0 : 1);
	
	
	checkCudaErrors(cudaMalloc((void**)&d_max_a, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_a, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_b, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_b, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_L, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_L, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_r, grid * sizeof(T)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_r, grid * sizeof(T)));
	checkCudaErrors(cudaMalloc((void**)&d_max_c, grid * sizeof(T)));
	checkCudaErrors(cudaMalloc((void**)&d_min_c, grid * sizeof(T)));

	minmax<T>(sp, d_max_a, d_min_a, d_aSp);
	minmax<T>(sp, d_max_b, d_min_b, d_bSp);
	minmax<T>(sp, d_max_L, d_min_L, d_lSp);
	minmax<T>(sp, d_max_r, d_min_r, d_rSp);
	minmax<T>(sp, d_max_c, d_min_c, d_cSp);

	NormalizeDataKernel << <numBlocks, numThreads >> > (sp, d_lSp, d_aSp, d_bSp, d_rSp, d_cSp, d_max_L, d_max_a, d_max_b, d_max_r, d_max_c, d_min_L,
		d_min_a, d_min_b, d_min_r, d_min_c
		);
	checkCudaErrors(cudaEventRecord(stop));
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double timeMS = milliseconds;
	printf("CreateSuperpixelAttributeGPUKern Final %f\n", timeMS);
	
	checkCudaErrors(cudaFree(d_max_a));
	checkCudaErrors(cudaFree(d_min_a));
	checkCudaErrors(cudaFree(d_max_b));
	checkCudaErrors(cudaFree(d_min_b));
	checkCudaErrors(cudaFree(d_max_L));
	checkCudaErrors(cudaFree(d_min_L));
	checkCudaErrors(cudaFree(d_max_r));
	checkCudaErrors(cudaFree(d_min_r));
	checkCudaErrors(cudaFree(d_max_c));
	checkCudaErrors(cudaFree(d_min_c));

	//return cudaPeekAtLastError();
}

template <typename T>
void CalculateSaliencyControlMapAndWeightsGpu(T *d_saliencyControlMap, T *d_weights, const  T *d_a, const  T *d_b, const  T *d_L, const  T *d_r, const  T *d_c, int numSuperpixels, bool locationPrior,
	const T r,const T sigma2) {
	int numThreads = 128;
	int numBlocks = (numSuperpixels / numThreads);
	int extra = (numSuperpixels % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	int choice = 1;
	
	// Launch a kernel on the GPU with one thread for each element.
	if (isPow2(numSuperpixels))
		CalculateSaliencyControlMapAndWeightsKernel<T,true> << < numSuperpixels, numSuperpixels, numSuperpixels * sizeof(T) >> > (d_saliencyControlMap, d_a, d_b, d_L, d_r, d_c, numSuperpixels, locationPrior, d_weights,sigma2,r);
	else		
		CalculateSaliencyControlMapAndWeightsKernel<T, false> << < numSuperpixels, numSuperpixels, numSuperpixels * sizeof(T) >> > (d_saliencyControlMap, d_a, d_b, d_L, d_r, d_c, numSuperpixels, locationPrior, d_weights, sigma2, r);
	
	if (choice == 0) { // Thrust
		T range;
		// setup arguments
		minmax_unary_op<T>  unary_op;
		minmax_binary_op<T> binary_op;
		//minmax_pair<T> result;
		// wrap raw pointer with a device_ptr 
		thrust::device_ptr<T> dev_ptr = thrust::device_pointer_cast(d_saliencyControlMap);
		// initialize reduction with the first value
		minmax_pair<T> init = unary_op(dev_ptr[0]);

		// compute minimum and maximum values
		minmax_pair<T> result = thrust::transform_reduce(dev_ptr, dev_ptr + numSuperpixels, unary_op, init, binary_op);

		//result.max_val = 3;
		//result.min_val = 2.1;
		range = result.max_val - result.min_val;
		NormalizeKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, range, result.min_val, numSuperpixels);

	}
	else {
		T *dev_max = NULL;
		T *dev_min = NULL;
		
		int blocksize = 128;
		int grid = (numSuperpixels / blocksize) + ((numSuperpixels % blocksize == 0) ? 0 : 1);
		
		checkCudaErrors(cudaMalloc((void**)&dev_max, grid * sizeof(T)));
	
		checkCudaErrors(cudaMalloc((void**)&dev_min, grid * sizeof(T)));
		
		
		minmax<T>(numSuperpixels, dev_max, dev_min, d_saliencyControlMap);
		NormalizeKernel_ << <numBlocks, numThreads >> > (d_saliencyControlMap, dev_min, dev_max, numSuperpixels);

		checkCudaErrors(cudaFree(dev_max));
		checkCudaErrors(cudaFree(dev_min));
	}
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double timeMS = milliseconds;
	printf("CalculateSaliencyControlMapAndWeightsGpu Final %f \n", timeMS);
	//return cudaPeekAtLastError();
}

template <typename T>
void CalculateKNNGpu(T *d_weights, int k, int sp) {



	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	switch (nextPow2(sp))
	{
	case 2:
		CalculateKNNKernel<T, 2> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 4:
		CalculateKNNKernel<T, 4> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 8:
		CalculateKNNKernel<T, 8> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;

	case 16:
		CalculateKNNKernel<T, 16> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 32:
		CalculateKNNKernel<T, 32> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 64:
		CalculateKNNKernel<T, 64> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 128:
		CalculateKNNKernel<T, 128> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 256:
		CalculateKNNKernel<T, 256> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 512:
		CalculateKNNKernel<T, 512> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	case 1024:
		CalculateKNNKernel<T, 1024> << <sp, sp, sp * sizeof(T) >> >(d_weights,k, sp);
		break;
	default:
		std::cout << "No possible conversion " << std::endl;
		break;
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double timeMS = milliseconds;
	printf("CalculateKNNGPUKern_ Final %f \n", timeMS);




	//return cudaPeekAtLastError();
}

// ## NLTVSalTerm ##
template <class T>
void NLTVSaliency(T* d_uk, T *d_bPD, T *d_pk, T *divP, const T *d_saliencyControlMap, const T * d_values, int *d_trans, const int * d_rows, const int * d_cols, const int numSuperpixels,
	const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol,const int salMethod,const cublasHandle_t& handlecublas) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int iter = 0;
#ifdef DEBUG_MATRIX	 
	MatrixMatFile matFile;
	
	
//	float * d_uk = (float*)malloc(sizeof(float) * numSuperpixels);
	float * h_uk = (float*)malloc(sizeof(float) * numSuperpixels);
	float *h_divP = (float*)malloc(sizeof(float)*numSuperpixels);
	float * h_pk =  (float*)malloc(sizeof(float) * knn * numSuperpixels);
#endif
	float a = ((deltaSalTerm*deltaSalTerm) / (alphaSalTerm *alphaSalTerm)) - (lambda / alphaSalTerm);
	T alpha2 = alphaSalTerm * alphaSalTerm;

	int numThreads = 128;
	int numBlocks = (numSuperpixels / numThreads);
	int extra = (numSuperpixels % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	int size = numSuperpixels*knn;
	int numBlocks2 = (size / numThreads);
	int extra2 = (size % numThreads == 0) ? 0 : 1;
	numBlocks2 = numBlocks2 + extra2;
	T divlambda = 1 / lambda;
	//printf("Numblocks %d threads %d \n", numBlocks, numThreads);
	//printf("Numblocks2 %d threads %d \n", numBlocks2, numThreads);
	cudaEventRecord(start);
	CalculateBKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_bPD, deltaSalTerm, alphaSalTerm, lambda, a, numSuperpixels);
	thrust::device_ptr<float> dev_ptr_pk = thrust::device_pointer_cast(d_pk);
	
	float *h_energy;
	float *d_energy;
	//
	float *d_max;
	
	int termSize = 3;
	h_energy = (float*)malloc(termSize * sizeof(float));
	checkCudaErrors(cudaMalloc((&d_energy), termSize * sizeof(float)));
	if (caseAlg == 0) {
		checkCudaErrors(cudaMalloc((&d_max), termSize * sizeof(float)));
	}
	
	
	
	// Starts with 1 because we divide in the gradient 
	float max = 1.0; 
	float energy;	
	

	CalculateTransIndexKernel << <numBlocks, numThreads >> > (d_trans, d_rows, d_cols, d_values, knn, numSuperpixels);
	energy = 10;
	float energyPrev = 10, energyDiff = 10;
	int blockReduction;
	
	bool exit = false;
	
	// http://www.cplusplus.com/reference/cmath/isnan/
	while (!exit && !isnan(energy)) {
	
		switch (caseAlg)
		{
		case 0: // Sal
			NLTVGradientKernel << <numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);

			Max<float,true> << <numSuperpixels, knn >> > (d_max, d_pk, size);
			//max = *thrust::max_element(dev_ptr_pk, dev_ptr_pk + size);
			//std::cout << "Iter " << iter << " Max " << max << std::endl;
		
			//if (max < 1.0f) max = 1.0f;
			
			//DivByMaxKernel << <numBlocks2, numThreads >> > (max, size, d_pk);
			DivByMaxKernel << <numSuperpixels, knn >> > (d_max, size, d_pk);
			
			NLTVDivKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_uk, d_pk, divP, knn, tau_d, numSuperpixels);
			// NLTV=0, NLTVSalTerm=1
			switch (salMethod) {
			case 0:
				UpdateVariableNLTVKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_uk, divP, tau_p, divlambda, numSuperpixels, d_energy);
				EnergyNLTVKernel << <numBlocks, numThreads >> > (numSuperpixels, d_uk, d_saliencyControlMap, d_rows, d_cols, d_values, lambda, d_energy);
				checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));

				h_energy[0] = h_energy[0];
				h_energy[1] = 0.5 *lambda *h_energy[1];
				energy = h_energy[0] + h_energy[1];
				break;
			case 1:
				UpdateVariableNLTVSalTermKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_uk, d_bPD, divP, tau_p, a, numSuperpixels, d_energy);
				EnergyNLTVSalTermKernel << <numBlocks, numThreads >> > (numSuperpixels, d_uk, d_saliencyControlMap, deltaSalTerm, d_energy);
				checkCudaErrors(cudaMemcpy(h_energy, d_energy, 3 * sizeof(*d_energy), cudaMemcpyDeviceToHost));

				h_energy[0] = alphaSalTerm*h_energy[0];
				h_energy[1] = 0.5 *lambda *h_energy[1];
				h_energy[2] = 0.5 * alpha2 * h_energy[2];

				energy = h_energy[0] + h_energy[1] - h_energy[2];
				break;
			}
			
			//std::cout << "Energy " << energy << " h_energy0 " << h_energy[0] << " h_energy1 " << h_energy[1] << " h_energy2 " << h_energy[2] << std::endl;
			break;
		 
		case 1: // SalPlus
			blockReduction = nextPow2(numSuperpixels) / 2;
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSaltermKernel<float, true> << <numSuperpixels,knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			else
				MaximizationNLTVSaltermKernel<float, false> << <numSuperpixels,knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
#ifdef DEBUG_MATRIX	 
		
			checkCudaErrors(cudaMemcpy(h_pk, d_pk, knn *numSuperpixels * sizeof(*d_pk), cudaMemcpyDeviceToHost));
			//matrixIOPk.store("d_pk_.txt", h_pk, numSuperpixels, knn);
			matFile.WriteOutputMatFile("pk_GPU.mat", "pk_GPU", h_pk, numSuperpixels, knn);
#endif

			// NLTV=0, NLTVSalTerm=1
			switch (salMethod) {
			case 0:
				if (isPow2(numSuperpixels))
					MinimizationNLTVEnergyKernel<T, true> << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, tau_p, divlambda, d_energy);
				else
					MinimizationNLTVEnergyKernel<T, false> << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, tau_p, divlambda, d_energy);

				checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));

				h_energy[0] = h_energy[0];
				h_energy[1] = 0.5 *lambda *h_energy[1];
				energy = h_energy[0] + h_energy[1];
				//std::cout << "Energy " << energy << " h_energy0 " << h_energy[0] << " h_energy1 " << h_energy[1] << std::endl;
				break;
			case 1:

				if (isPow2(numSuperpixels))
					MinimizationNLTVSalTermEnergyKernel<float, true> << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, d_bPD, tau_p, a, d_energy, deltaSalTerm);
				else
					MinimizationNLTVSalTermEnergyKernel<float, false> << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, d_bPD, tau_p, a, d_energy, deltaSalTerm);


				checkCudaErrors(cudaMemcpy(h_energy, d_energy, 3 * sizeof(*d_energy), cudaMemcpyDeviceToHost));
				h_energy[0] = 1 * h_energy[0];
				h_energy[1] = 0.5 * alphaSalTerm * lambda  *  h_energy[1];
				h_energy[2] = 0.5 * alpha2 * h_energy[2];

				energy = h_energy[0] + h_energy[1] - h_energy[2];
				//std::cout << "Energy " << energy << " h_energy0 " << h_energy[0] << " h_energy1 " << h_energy[1] << " h_energy2 " << h_energy[2] << std::endl;
				break;
			}

#ifdef DEBUG_MATRIX	 
			checkCudaErrors(cudaMemcpy(h_uk, d_uk, numSuperpixels * sizeof(*d_uk), cudaMemcpyDeviceToHost));
			//matrixIOUk.store("d_uk.txt", h_uk, 1, numSuperpixels);
			matFile.WriteOutputMatFile("uk_GPU.mat", "uk_GPU", h_uk, 1, numSuperpixels);
#endif
			break;

		case 2: // SalStar
			//blockReduction = nextPow2(numSuperpixels) / 2;
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSaltermKernel<float, true> << <numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			else
				MaximizationNLTVSaltermKernel<float, false> << <numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			// NLTV=0, NLTVSalTerm=1
			switch (salMethod) {
			case 0:
				MinimizationNLTVNoEnergyKernel << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, tau_p, divlambda);
				break;
			case 1:
				MinimizationNLTVSalTermNoEnergyKernel<float> << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, d_bPD, tau_p, a);
				break;

			}
				
			break;
		default:
			break;
		}
		
		if (max < 1.0f) max = 1.0f;
		if (iter == 0) {
			energyPrev = energy;
			//std::cout <<" energyPrev " << energyPrev<< std::endl;
		}
		else {
			energyDiff = sqrt((energy - energyPrev) * (energy - energyPrev));
			energyPrev = energy;
			//std::cout <<"iter "<< iter<< "energy "<< energy << " energyPrev " << energyPrev << "Energy Diff " << energyDiff << std::endl;
		}

		iter++;

		if (tol == 0) {
			exit = iter >= maxIter;
		}
		else {
			exit = energyDiff < tol;

		}
	}	
	
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double timeMS = milliseconds;
	if (salMethod==0) {
		printf("IterativeNLTVGpu Final %f Iter completed %i Final Energy %f\n", timeMS, iter, energyDiff);
	}
	else {
		printf("IterativeNLTVSaliencyTermGpu Final %f Iter completed %i Final Energy %f\n", timeMS, iter, energyDiff);
	}
#ifdef DEBUG_MATRIX	 
	
	free(h_divP);
	free(h_pk);
	free(h_uk);
#endif		
	checkCudaErrors(cudaFree(d_energy));
	if (caseAlg == 0) {
		checkCudaErrors(cudaFree(d_max));		
	}
	
	free(h_energy);

	
	//return cudaPeekAtLastError();
}

template <typename T>
void NLTVGpu(T* d_uk, T *b, T *d_pk, T *divP, const T *d_saliencyControlMap, const T * d_values, int *d_trans, const int * d_rows, const int * d_cols, int numSuperpixels,
	 const T lambda, const T tau_d, const T tau_p, const int knn, const int maxIter, const int caseAlg, const T tol, const cublasHandle_t& handlecublas) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	int iter = 0;

	int numThreads = 128;
	int numBlocks = (numSuperpixels / numThreads);
	int extra = (numSuperpixels % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	int size = numSuperpixels*knn;
	int numBlocks2 = (size / numThreads);
	int extra2 = (size % numThreads == 0) ? 0 : 1;
	numBlocks2 = numBlocks2 + extra2;

	//printf("Numblocks %d threads %d \n", numBlocks, numThreads);
	//printf("Numblocks2 %d threads %d \n", numBlocks2, numThreads);
	cudaEventRecord(start);
	
	thrust::device_ptr<T> dev_ptr_pk = thrust::device_pointer_cast(d_pk);
	
	T *h_energy;
	T *d_energy;
	T *d_max;

	int termSize = 2;
	h_energy = (T*)malloc(termSize * sizeof(T));
	checkCudaErrors(cudaMalloc((&d_energy), termSize * sizeof(T)));
	if (caseAlg == 0) {
		checkCudaErrors(cudaMalloc((&d_max), numSuperpixels * sizeof(float)));
	}

	// Starts with 1 because we divide in the gradient 
	T max = 1.0;
	T divlambda = 1 / lambda;

	T energy;	
	

	CalculateTransIndexKernel << <numBlocks, numThreads >> > (d_trans, d_rows, d_cols, d_values,  knn, numSuperpixels);
	energy = 10;
	T energyPrev = 10, energyDiff = 10;
	
	
#ifdef DEBUG_MATRIX	 
	
	float *h_divP = (float*)malloc(sizeof(float)*numSuperpixels);
#endif
	
	float * h_max = (float*)(malloc(sizeof(float)*knn*numSuperpixels));
	bool exit = false;
	// http://www.cplusplus.com/reference/cmath/isnan/
	while (!exit && !isnan(energy)) {

		switch (caseAlg)
		{
		case 0: // Sal NLTV
			NLTVGradientKernel << <numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			//MaximizationNLTVSaltermKernel<T, true> << < numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			Max<float, true> << <numSuperpixels, knn >> > (d_max, d_pk, size);
		
#ifdef DEBUG_MATRIX	 
			
			checkCudaErrors(cudaMemcpy(h_max, d_pk,knn* numSuperpixels * sizeof(float), cudaMemcpyDeviceToHost));
			matrixIOMax.store("h_max.txt", h_max, numSuperpixels, knn);
			//free(h_max);
#endif
			DivByMaxKernel << <numSuperpixels, knn >> > (d_max, size, d_pk);

			NLTVDivKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_uk, d_pk, divP, knn, tau_d, numSuperpixels);
#ifdef DEBUG_MATRIX	 
			checkCudaErrors(cudaMemcpy(h_divP, divP, numSuperpixels * sizeof(*divP), cudaMemcpyDeviceToHost));
			matrixIO.store("divPGPU.txt", h_divP, 1, numSuperpixels);
#endif
			UpdateVariableNLTVKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, d_uk, divP, tau_p, divlambda, numSuperpixels, d_energy);
#ifdef DEBUG_MATRIX	 
			checkCudaErrors(cudaMemcpy(h_divP, d_uk, numSuperpixels * sizeof(*divP), cudaMemcpyDeviceToHost));
			matrixIO.store("ukGPU.txt", h_divP, 1, numSuperpixels);
#endif
			EnergyNLTVKernel << <numBlocks, numThreads >> > (numSuperpixels, d_uk, d_saliencyControlMap, d_rows, d_cols, d_values, lambda, d_energy);
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));
			
			h_energy[0] = h_energy[0];
			h_energy[1] = 0.5 *lambda *h_energy[1];
			energy = h_energy[0] + h_energy[1] ;
			//std::cout << "Energy " << energy << " h_energy0 " << h_energy[0] << " h_energy1 " << h_energy[1] << std::endl;
		
			break;
		
		case 1: // SalPlus NLTV
			
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSaltermKernel<T, true> << < numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			else
				MaximizationNLTVSaltermKernel<T, false> << < numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			if (isPow2(numSuperpixels))
				MinimizationNLTVEnergyKernel<T, true> << <1, numSuperpixels  >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk,  tau_p, divlambda, d_energy);
			else
				MinimizationNLTVEnergyKernel<T, false> << <1, numSuperpixels>> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, tau_p, divlambda, d_energy);
			
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));
		
			h_energy[0] = h_energy[0];
			h_energy[1] = 0.5 *lambda *h_energy[1];
			energy = h_energy[0] + h_energy[1];
			//std::cout << "Energy " << energy << " h_energy0 " << h_energy[0] << " h_energy1 " << h_energy[1] << std::endl;
			break;

		case 2: // SalStar NLTV
					
			if (isPow2(numSuperpixels))
				MaximizationNLTVSaltermKernel<T, true> << <numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);
			else
				MaximizationNLTVSaltermKernel<T, false> << < numSuperpixels, knn >> > (d_rows, d_cols, d_values, d_uk, d_pk, knn, tau_d, size);

			MinimizationNLTVNoEnergyKernel << <1, numSuperpixels >> > (d_saliencyControlMap, d_rows, d_cols, d_values, d_trans, d_pk, knn, tau_d, numSuperpixels, d_uk, tau_p, divlambda);
			
			break;
		default:
			break;
		}


		if (max < 1.0f) max = 1.0f;
		if (iter == 0) {
			energyPrev = energy;
			//std::cout <<" energyPrev " << energyPrev<< std::endl;
		}
		else {
			energyDiff = sqrt((energy - energyPrev) * (energy - energyPrev));
			energyPrev = energy;
			//std::cout <<"iter "<< iter<< "energy "<< energy << " energyPrev " << energyPrev << "Energy Diff " << energyDiff << std::endl;
		}

		iter++;

		if (tol == 0) {
			exit = iter >= maxIter;
		}
		else {
			exit = energyDiff < tol;

		}
	}	

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	double timeMS = milliseconds;
	printf("IterativeNLTV Final %f Iter completed %i Final Energy %f\n", timeMS, iter, energyDiff);
#ifdef DEBUG_MATRIX	 
	checkCudaErrors(cudaMemcpy(h_divP, d_uk, numSuperpixels * sizeof(*divP), cudaMemcpyDeviceToHost));
	matrixIO.store("_v_k1GPU.txt", h_divP, 1, numSuperpixels);
	free(h_divP);
#endif	
	checkCudaErrors(cudaFree(d_energy));
	if (caseAlg == 0) {
		checkCudaErrors(cudaFree(d_max));
	}
	free(h_energy);
	
	
	//return cudaPeekAtLastError();
}

template void CreateSuperpixelAttributeGpu<float>(int height, int width, const int *d_labels, const float *d_L, const float* d_a, const float* d_b, const int sp, float *d_LSp, float  *d_aSp, float *d_bSp, float *d_rSp, float *d_cSp, int *numSp);

template void
CalculateSaliencyControlMapAndWeightsGpu<float>(float *d_saliencyControlMap, float *d_weights, const  float *d_a, const  float *d_b, const  float *d_L, const  float *d_r, const  float *d_c, int sp, bool locationPrior, const float r, const float sigma2);

template void
CalculateKNNGpu<float>(float *d_weights, int k, int sp);
template void
NLTVSaliency<float>(float* d_uk, float *d_bPD, float *d_pk, float *divP, const float *d_saliencyControlMap, const float * d_values, int *d_trans, const int * d_rows, const int * d_cols, int _N, const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol, const int salMethod, const cublasHandle_t& handlecublas);


template void
RgbToLabGpu<float>(const int* d_rin, const int* d_gin, const int* d_bin, const int sz, float* d_L, float* d_a, float* d_b);

#endif // __NLTV_Saliency_Alg_Kernel_CU__