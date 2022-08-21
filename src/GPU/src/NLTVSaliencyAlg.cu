
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "MyUtil.h"
#include "MimMax.h"
#include "helper_cuda.h"
#include <cublas_v2.h>
#include <stdio.h>

// https://devblogs.nvidia.com/cuda-pro-tip-optimized-filtering-warp-aggregated-atomics/
// https://github.com/thrust/thrust/blob/master/examples/minmax.cu

cudaError_t CreateSuperpixelAttributeGpu(int height, int width, const int *d_labels, const float *d_L, const float* d_a, const float* d_b, const int sp, float *d_LSp, float  *d_aSp, float *d_bSp, float *d_rSp, float *d_cSp, int *numSp);
__host__
cudaError_t NLTVSaliencyTermGpu(float* _v_k1, float *b, float *pk, float *divP, const float *saliencyControlMap, const float * csrVal, int *csrValII, const int * csrRow, const int * csrCol, int _N, const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter,const int caseAlg,const float tol, const cublasHandle_t& handlecublas);

__host__
cudaError_t NLTVGpu(float* uk, float *b, float *pk, float *divP, const float *saliencyControlMap, const float * csrVal, int *csrValII, const int * csrRow, const int * csrCol, int numSuperpixels,
	const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol, const cublasHandle_t& handlecublas);

__host__
cudaError_t CalculateKNNGpu(float *d_weights, int k, int width, int height);

__host__
cudaError_t RgbToLabGpu(const int* rin, const int* gin, const int* bin, const int sz, float* lvec, float* avec, float* bvec);


// https://devblogs.nvidia.com/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/
__global__ void RgbToLabKernel(const int* rin, const int* gin, const int* bin, const int sz, float* lvec, float* avec, float* bvec) {
	
	for (int i = blockIdx.x * blockDim.x + threadIdx.x;
		i < sz;
		i += blockDim.x * gridDim.x)
	{

		int sR, sG, sB;
		float R, G, B;
		float X, Y, Z;
		float r, g, b;
		const float epsilon = 0.008856f;	//actual CIE standard
		const float kappa = 903.3f;		//actual CIE standard

		const float Xr = 0.950456f;	//reference white
		const float Yr = 1.0f;		//reference white
		const float Zr = 1.088754f;	//reference white
		float xr, yr, zr;
		float fx, fy, fz;
		float lval, aval, bval;


		sR = rin[i]; sG = gin[i]; sB = bin[i];
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

		lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
	}
}

template <typename Dtype>
__global__ void CreateSuperpixelAttributeKernel(int n,  const int *d_labels, const Dtype *d_l, const Dtype* d_a, const Dtype* d_b, const int outputNumSuperpixels, Dtype *d_lSp, Dtype  *d_aSp, Dtype *d_bSp, Dtype *d_rSp, Dtype *d_cSp, int *d_numSp) {
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

template <typename Dtype>
__global__ void DivideNumSuperpixelKernel(int n, Dtype *lvec, Dtype* avec, Dtype* bvec, Dtype *rValue, Dtype *cValue, int *numSp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n && numSp[i]>0) {		
		lvec[i] = lvec[i] / (float)numSp[i];
		avec[i] = avec[i] / (float)numSp[i];
		bvec[i] = bvec[i] / (float)numSp[i];
		rValue[i] = rValue[i] / (float)numSp[i];
		cValue[i] = cValue[i] / (float)numSp[i];
	}
}

template <typename Dtype>
__global__ void NormalizeDataKernel(int n, Dtype *lvec, Dtype* avec, Dtype* bvec, Dtype *rValue, Dtype *cValue, Dtype *maxValueL_, Dtype *maxValuea_, Dtype *maxValueb_, Dtype * maxValuer_, Dtype *maxValuec_,
	Dtype *minValueL_, Dtype *minValuea_, Dtype *minValueb_, Dtype *minValuer_, Dtype *minValuec_) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n) {
		
		lvec[i] = (lvec[i] - *minValueL_) / (*maxValueL_ - *minValueL_);
		avec[i] = (avec[i] - *minValuea_) / (*maxValuea_ - *minValuea_);
		bvec[i] = (bvec[i] - *minValueb_) / (*maxValueb_ - *minValueb_);
		rValue[i] = (rValue[i] - *minValuer_) / (*maxValuer_ - *minValuer_);
		cValue[i] = (cValue[i] - *minValuec_) / (*maxValuec_ - *minValuec_);

	}
}


template <typename Dtype,bool nIsPow2>
__global__ void CalculateSaliencyControlMapAndWeightsKernel(Dtype *d_saliencyControlMap, const  Dtype *d_a, const  Dtype *d_b, const  Dtype *d_L, const  Dtype *d_r, const  Dtype *d_c, int numSuperpixels, bool locationPrior,  Dtype *d_weights, const Dtype sigma2,const Dtype r) {
	int i = blockIdx.x;
	int j = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	
	Dtype v_p_obj = 0, w_pq = 0;
	
	Dtype sum;
	Dtype l_i_r, l_i_c, l_j_r, l_j_c, a_i, b_i, L_i;
	Dtype aux;
	// Normalize
	l_i_r = d_r[i];
	l_i_c = d_c[i];
	a_i = d_a[i];
	b_i = d_b[i];
	L_i = d_L[i];
	Dtype  l_bar_r, l_bar_c;
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
	sum = w_pq* aux;
	sum = blockReduceSum(sum, nIsPow2);
	/** Contrast prior PART */

	/** WEIGHTS PART */

	//Dtype alpha = 0.9;

	Dtype aAux = r*d_a[i];
	Dtype bAux = r*d_b[i];
	Dtype LAux = r*d_L[i];
	Dtype rAux = d_r[i];
	Dtype cAux = d_c[i];


	Dtype aAuxII = r* d_a[j];
	Dtype bAuxII = r* d_b[j];
	Dtype LAuxII = r* d_L[j];
	Dtype exponentValue = (aAux - aAuxII) * (aAux - aAuxII) +
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
		
	}
	/** Calculation Control Map for j superpixel ends */

}

template <typename Dtype>
__global__ void NormalizeKernel(Dtype *saliencyControlMap, Dtype range, Dtype minValue, const int numSuperixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperixels) {
		saliencyControlMap[i] = (saliencyControlMap[i] - minValue) / range;
	}
}

template <typename Dtype>
__global__ void NormalizeKernel_(Dtype *saliencyControlMap, const  Dtype *minValue, const Dtype *maxValue, const int numSuperixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperixels) {
		Dtype range = *maxValue - *minValue;
		saliencyControlMap[i] = (saliencyControlMap[i] - *minValue) / range;
	}
}

__host__ 
cudaError_t RgbToLabGpu(const int* rin, const int* gin,const int* bin,const int sz, float* lvec, float* avec, float* bvec) {
	int numThreads = 128;
	int numBlocks = (sz / numThreads);
	int extra = (sz % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	RgbToLabKernel << <numBlocks, numThreads >> > (rin, gin, bin, sz, lvec, avec, bvec);

	return cudaGetLastError();
}
//
__host__
cudaError_t CreateSuperpixelAttributeGpu(int height, int width, const int *d_labels, const float *d_l, const float* d_a, const float* d_b, const int sp, float *d_lSp, float  *d_aSp, float *d_bSp, float *d_rSp, float *d_cSp, int *numSp) {
	int n = height *width;
	float *d_max_a = 0;
	float *d_min_a = 0;
	float *d_min_b = 0;
	float *d_max_b = 0;
	float *d_min_L = 0;
	float *d_max_L = 0;
	float *d_min_r = 0;
	float *d_max_r = 0;
	float *d_min_c = 0;
	float *d_max_c = 0;

	
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
	
	
	checkCudaErrors(cudaMalloc((void**)&d_max_a, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_a, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_b, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_b, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_L, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_L, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_max_r, grid * sizeof(float)));
	
	checkCudaErrors(cudaMalloc((void**)&d_min_r, grid * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_max_c, grid * sizeof(float)));
	checkCudaErrors(cudaMalloc((void**)&d_min_c, grid * sizeof(float)));

	minmax<float>(sp, d_max_a, d_min_a, d_aSp);
	minmax<float>(sp, d_max_b, d_min_b, d_bSp);
	minmax<float>(sp, d_max_L, d_min_L, d_lSp);
	minmax<float>(sp, d_max_r, d_min_r, d_rSp);
	minmax<float>(sp, d_max_c, d_min_c, d_cSp);

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

	return cudaPeekAtLastError();
}


__host__
cudaError_t CalculateSaliencyControlMapAndWeightsGpu(float *d_saliencyControlMap, float *d_weights, const  float *d_a, const  float *d_b, const  float *d_L, const  float *d_r, const  float *d_c, int numSuperpixels, bool locationPrior,
	const float r,const float sigma2) {
	int numThreads = 128;
	int numBlocks = (numSuperpixels / numThreads);
	int extra = (numSuperpixels % numThreads == 0) ? 0 : 1;
	numBlocks = numBlocks + extra;
	
	int choice = 1;
	
	// Launch a kernel on the GPU with one thread for each element.
	if (isPow2(numSuperpixels))
		CalculateSaliencyControlMapAndWeightsKernel<float,true> << < numSuperpixels, numSuperpixels, numSuperpixels * sizeof(float) >> > (d_saliencyControlMap, d_a, d_b, d_L, d_r, d_c, numSuperpixels, locationPrior, d_weights,sigma2,r);
	else 
		CalculateSaliencyControlMapAndWeightsKernel<float, true> << < numSuperpixels, numSuperpixels, numSuperpixels * sizeof(float) >> > (d_saliencyControlMap, d_a, d_b, d_L, d_r, d_c, numSuperpixels, locationPrior,d_weights,sigma2, r);

	if (choice == 0) { // Thrust
		float range;
		// setup arguments
		minmax_unary_op<float>  unary_op;
		minmax_binary_op<float> binary_op;
		//minmax_pair<float> result;
		// wrap raw pointer with a device_ptr 
		thrust::device_ptr<float> dev_ptr = thrust::device_pointer_cast(d_saliencyControlMap);
		// initialize reduction with the first value
		minmax_pair<float> init = unary_op(dev_ptr[0]);

		// compute minimum and maximum values
		minmax_pair<float> result = thrust::transform_reduce(dev_ptr, dev_ptr + numSuperpixels, unary_op, init, binary_op);

		//result.max_val = 3;
		//result.min_val = 2.1;
		range = result.max_val - result.min_val;
		NormalizeKernel << <numBlocks, numThreads >> > (d_saliencyControlMap, range, result.min_val, numSuperpixels);

	}
	else {
		float *dev_max = NULL;
		float *dev_min = NULL;
		
		int blocksize = 128;
		int grid = (numSuperpixels / blocksize) + ((numSuperpixels % blocksize == 0) ? 0 : 1);
		
		checkCudaErrors(cudaMalloc((void**)&dev_max, grid * sizeof(float)));
	
		checkCudaErrors(cudaMalloc((void**)&dev_min, grid * sizeof(float)));
		
		
		minmax<float>(numSuperpixels, dev_max, dev_min, d_saliencyControlMap);
		NormalizeKernel_ << <numBlocks, numThreads >> > (d_saliencyControlMap, dev_min, dev_max, numSuperpixels);

		checkCudaErrors(cudaFree(dev_max));
		checkCudaErrors(cudaFree(dev_min));
	}
	
	return cudaPeekAtLastError();
}

template <typename Dtype, unsigned int blockSize>
__global__ void CalculateKNNKernel(Dtype* d_weights, const int width) {

	//int i = blockIdx.x;
	int j = threadIdx.x;
	int index = blockIdx.x * blockDim.x + threadIdx.x;

	Dtype *sdata = SharedMemory<Dtype>();
	__shared__ int maxsFinal[5];
	__shared__ int maxs[blockSize];


	// do reduction in shared mem
	for (int h = 0; h < 5; h++) {
		sdata[j] = d_weights[index];
		maxs[j] = j;

		__syncthreads();
		for (int k = 0; k < h; k++) {
			sdata[maxsFinal[k]] = 0;
		}
		__syncthreads();
		for (unsigned int s = blockSize/2; s > 0; s >>= 1)
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
		__syncthreads();
	}
	if (j != maxsFinal[0] && j != maxsFinal[1] && j != maxsFinal[2] && j != maxsFinal[3] && j != maxsFinal[4]) {
		
		d_weights[index] = 0.0;
	}
}



__host__
cudaError_t CalculateKNNGpu(float *d_weights, int k, int width, int height) {

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	switch (nextPow2(width))
	{
	case 2:
		CalculateKNNKernel<float, 2> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 4:
		CalculateKNNKernel<float, 4> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 8:
		CalculateKNNKernel<float, 8> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;

	case 16:
		CalculateKNNKernel<float, 16> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 32:
		CalculateKNNKernel<float, 32> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 64:
		CalculateKNNKernel<float, 64> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 128:
		CalculateKNNKernel<float, 128> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 256:
		CalculateKNNKernel<float, 256> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 512:
		CalculateKNNKernel<float, 512> << <width, width, width * sizeof(float) >> >(d_weights, width);
		break;
	case 1024:
		CalculateKNNKernel<float, 1024> << <width, width, width * sizeof(float) >> >(d_weights, width);
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
	return cudaPeekAtLastError();
}

template <typename Dtype>
__global__ void CalculateB(const  Dtype *saliencyControlMap, Dtype *b, const Dtype  deltaSalTerm, const Dtype alphaSalTerm, const Dtype lambda, const Dtype a, int numSuperpixels) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixels) {
		b[i] = (deltaSalTerm / (alphaSalTerm * alphaSalTerm)) - (lambda / alphaSalTerm)*saliencyControlMap[i];
	}
}

template <typename Dtype>
__global__ void DivByMax(const Dtype max, const int N, Dtype *pk) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		pk[i] = pk[i] / max;
	}
}

template <typename Dtype>
__global__ void CalculateTransIndex(int *csrValII, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const Dtype * pMatrix, const int knn, const int N) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < N) {
		int aux = p* knn;		
		int row_start = csrRow[p];
		int row_end = csrRow[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int row = csrCol[jj];
			int col = p;
			int val = -1;
			for (int h = csrRow[row], z = 0; csrCol[h] <= col && h < csrRow[row + 1]; h++, z++) {
				if (csrCol[h] == col) {
					val = z;
					break;
				}
			}
			csrValII[aux + k] = val;
		}
	}
}

template <typename Dtype>
__global__ void NLTVGradient(const int  *csrRow, const int * csrCol, const Dtype * csrVal, const Dtype * uk, Dtype * pk, const int k, const Dtype tau_d, int _N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < _N) {
		int index = i / k;
		int col = i % k;
		int row_start = csrRow[index];
		int row_end = csrRow[index + 1];

		if ((row_end - row_start) > col) {
			int jj = row_start + col;
			pk[i] = pk[i] + tau_d * (uk[csrCol[jj]] - uk[index])* sqrt(csrVal[jj]);
		}
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


template <typename Dtype, bool nIsPow2>
__global__ void MaximizationNLTVSalterm(const int  *csrRow, const int * csrCol, const Dtype * csrVal, const Dtype * uk, Dtype * pk, const int k, const Dtype tau_d, int _N, Dtype *max, int blockReduction, int iter, int itertoWrite) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = threadIdx.x;
	Dtype maxValue=0;
	if (i < _N) {
		int index = i / k;
		int col = i - (index * k);
		//int col = i % k;
		int row_start = csrRow[index];
		int row_end = csrRow[index + 1];
		Dtype aux = pk[i] / max[iter];
		if ((row_end - row_start) > col) {


			int jj = row_start + col;

			pk[i] = aux + tau_d * (uk[csrCol[jj]] - uk[index])* sqrt(csrVal[jj]);
			aux = pk[i];
		}
		maxValue = aux;
		maxValue = blockReduceMax(maxValue, nIsPow2);
		if (0 == j) {
			if (maxValue > 1.0) {
				MyatomicMax(&max[itertoWrite], maxValue);
			}
			max[3 - itertoWrite - iter] = 1.0;
		}
	}
}

template <typename Dtype>
__global__ void NLTVDiv(const  Dtype *saliencyControlMap, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const Dtype * uk, const Dtype * pMatrix, Dtype *divP, const int knn, const Dtype tau_d, const int N) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	//int p = i / knn;
	if (p < N) {
		int aux = p* knn;
		Dtype sumValue = 0.0;
		int row_start = csrRow[p];
		int row_end = csrRow[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int row = csrCol[jj];
			int col = p;
			float val = 0.0;
			for (int h = csrRow[row], z = 0; csrCol[h] <= col && h < csrRow[row + 1]; h++, z++) {
				if (csrCol[h] == col) {
					val = pMatrix[row*knn + z];
					break;
				}
			}
			sumValue = sumValue + ( pMatrix[aux + k] - val)* sqrt(csrVal[jj]);
		}
		divP[p] = sumValue;
	}
}

template <typename Dtype, bool nIsPow2>
__global__ void MinimizationNLTVSalTermEnergy(const  Dtype *saliencyControlMap, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const int * csrValII, const Dtype * pMatrix, const int knn, const Dtype tau_d, const int N, Dtype * uk, const Dtype * b, const Dtype tau_p, const Dtype a, Dtype *term, Dtype *max, const  int iter, const int blockReduction, const Dtype delta) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	Dtype sum;
	if (p < N) {

		Dtype sumValue = 0.0;
		int row_start = csrRow[p];
		int row_end = csrRow[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
								
			int index = p*knn + k;
			Dtype val = 0.0;
			if (csrValII[index] != -1) {
				int row = csrCol[jj];
				int col = csrValII[index];
				val = (pMatrix[row*knn + col]) / max[iter];
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ((pMatrix[p* knn + k] - val) / max[iter])* sqrt(csrVal[jj]);
		}
		//Dtype divP = sumValue;
		sumValue = (1 + a * tau_p) * uk[p] + tau_p *(sumValue - b[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;
		
		sum = (1 - delta*sumValue)*(1 - delta*sumValue);
		sum = blockReduceSum(sum, nIsPow2);

		if (p == 0) term[1] = sum;

		// (uk[p] - saliencyControlMap[p])*(uk[p] - saliencyControlMap[p]);
		sum = (sumValue - saliencyControlMap[p])*(sumValue - saliencyControlMap[p]);
		sum = blockReduceSum(sum, nIsPow2);
		if (p == 0) term[2] = sum;

		uk[p] = sumValue;

		__syncthreads();
		sumValue = 0.0;
		for (int jj = row_start; jj < row_end; jj++) {

			int q = csrCol[jj];
			sumValue = sumValue + (uk[q] - uk[p]) * (uk[q] - uk[p])* csrVal[jj];
		}
		sumValue = sqrt(sumValue);
		sumValue = blockReduceSum(sumValue, nIsPow2);
		term[0] = sumValue;
	}
}


template <typename Dtype>
__global__ void MinimizationNLTVSalTermNoEnergy(const  Dtype *saliencyControlMap, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const int * csrValII, const Dtype * pMatrix, const int knn, const Dtype tau_d, const int N, Dtype * uk, const Dtype * b, const Dtype tau_p, const Dtype a, Dtype *max, const  int iter) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;	
	if (p < N) {

		Dtype sumValue = 0.0;
		int row_start = csrRow[p];
		int row_end = csrRow[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int index = p*knn + k;
			Dtype val = 0.0;
			if (csrValII[index] != -1) {
				int row = csrCol[jj];
				int col = csrValII[index];
				val = (pMatrix[row*knn + col]) / max[iter];
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ((pMatrix[p* knn + k] - val) / max[iter])* sqrt(csrVal[jj]);
		}
		
		sumValue = (1 + a * tau_p) * uk[p] + tau_p *(sumValue - b[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;
		uk[p] = sumValue;	
	}
}

template <typename Dtype, bool nIsPow2 >
__global__ void MinimizationNLTVEnergy(const  Dtype *saliencyControlMap, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const int * csrValII, const Dtype * pMatrix, const int knn, const Dtype tau_d, const int N, Dtype * uk, const Dtype tau_p, const Dtype divA, Dtype *term, Dtype *max, const  int iter, const int blockReduction) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	Dtype sum;
	
	if (p < N) {

		Dtype sumValue = 0.0;
		int row_start = csrRow[p];
		int row_end = csrRow[p + 1];
		for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {
			int index = p*knn + k;
			Dtype val = 0.0;
			if (csrValII[index] != -1) {
				int row = csrCol[jj];
				int col = csrValII[index];
				val = (pMatrix[row*knn + col]) / max[iter];
			}
			//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
			sumValue = sumValue + ( (pMatrix[p* knn + k]) / max[iter] - val )* sqrt(csrVal[jj]);
		}		
		sumValue = (1 - tau_p) * uk[p] + tau_p *(divA *sumValue + saliencyControlMap[p]);
		if (sumValue < 0.0)
			sumValue = 0.0;
		else if (sumValue > 1.0)
			sumValue = 1.0;
		
		// (uk[p] - saliencyControlMap[p])*(uk[p] - saliencyControlMap[p]);
		sum = (sumValue - saliencyControlMap[p])*(sumValue - saliencyControlMap[p]);
		sum = blockReduceSum(sum, nIsPow2);
		
		if (p == 0) term[1] = sum;

		uk[p] = sumValue;		
		__syncthreads();
		sumValue = 0.0;
		for (int jj = row_start; jj < row_end; jj++) {

			int q = csrCol[jj];
			sumValue = sumValue + (uk[q] - uk[p]) * (uk[q] - uk[p])* csrVal[jj];
		}
		sumValue = sqrt(sumValue);
		sumValue = blockReduceSum(sumValue, nIsPow2);
		term[0] = sumValue;
	}
}

template <typename Dtype>
	__global__ void MinimizationNLTVNoEnergy(const  Dtype *saliencyControlMap, const int  *csrRow, const int * csrCol, const Dtype * csrVal, const int * csrValII, const Dtype * pMatrix, const int knn, const Dtype tau_d, const int N, Dtype * uk, const Dtype tau_p, const Dtype divA,  Dtype *max, const  int iter) {
		int p = blockIdx.x * blockDim.x + threadIdx.x;

		if (p < N) {

			Dtype sumValue = 0.0;
			int row_start = csrRow[p];
			int row_end = csrRow[p + 1];
			for (int jj = row_start, k = 0; jj < row_end; jj++, k++) {	
				int index = p*knn + k;
				Dtype val = 0.0;
				if (csrValII[index] != -1) {
					int row = csrCol[jj];
					int col = csrValII[index];
					val = (pMatrix[row*knn + col]) / max[iter];
				}
				//   sumValue  = sumValue + (pMatrix(p,q)-pMatrix(q,p))* sqrt (w(p,q));
				sumValue = sumValue + ((pMatrix[p* knn + k]) / max[iter] - val)* sqrt(csrVal[jj]);
			}
			//Dtype divP = sumValue;
			sumValue = (1 - tau_p) * uk[p] + tau_p *(divA *sumValue + saliencyControlMap[p]);
			if (sumValue < 0.0)
				sumValue = 0.0;
			else if (sumValue > 1.0)
				sumValue = 1.0;		
			uk[p] = sumValue;			
		}
	}

template <typename Dtype>
__global__ void UpdateVariableNLTVSalTerm(const  Dtype *saliencyControlMap, Dtype * uk, const Dtype * b, const Dtype *divP, const Dtype tau_p, const Dtype a, const int numSuperpixel,Dtype * d_term) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixel) {
		
		Dtype aux = (1 + a *tau_p) * uk[i] + tau_p *(divP[i] - b[i]);
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


template <typename Dtype>
__global__ void UpdateVariableNLTV(const  Dtype *saliencyControlMap, Dtype * uk, const Dtype *divP, const Dtype tau_p, const Dtype divLambda, const int numSuperpixel, Dtype * d_term) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < numSuperpixel) {
		//uk[i] = (1 - tau_p) * uk[i] + tau_p *(divA *divP[i] + b[i]);
		Dtype aux = (1 - tau_p) * uk[i] + tau_p *(divLambda* divP[i] + saliencyControlMap[i]);
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

template <typename Dtype>
__global__ void energyNLTVSalTerm(const int _N, const Dtype * uk, const Dtype * saliencyControlMap, const Dtype alpha, const Dtype delta, Dtype lambda, Dtype *term) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < _N) {
		Dtype term1Aux = 0.0;
		for (int q = 0; q < _N; q++) {

			term1Aux = term1Aux + ((uk[q] - uk[p]) * (uk[q] - uk[p]));
		}

		atomicAdd(&term[0], sqrt(term1Aux));
		atomicAdd(&term[1], (1 - delta* uk[p])*(1 - delta* uk[p]));
		atomicAdd(&term[2], (uk[p] - saliencyControlMap[p])*(uk[p] - saliencyControlMap[p]));
	}
}

template <typename Dtype>
__global__ void energyNLTV(const int _N, const Dtype * uk, const Dtype * saliencyControlMap, Dtype lambda, Dtype *term) {
	int p = blockIdx.x * blockDim.x + threadIdx.x;
	if (p < _N) {
		Dtype term1Aux = 0.0;
		for (int q = 0; q < _N; q++) {

			term1Aux = term1Aux + ((uk[q] - uk[p]) * (uk[q] - uk[p]));
		}
		atomicAdd(&term[0], sqrt(term1Aux));		
		atomicAdd(&term[1], (uk[p] - saliencyControlMap[p])*(uk[p] - saliencyControlMap[p]));
	}
}



// ## NLTVSalTerm ##
__host__
cudaError_t NLTVSaliencyTermGpu(float* uk, float *b, float *pk, float *divP, const float *saliencyControlMap, const float * csrVal, int *csrValII, const int * csrRow, const int * csrCol, int numSuperpixels,
	const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol,const cublasHandle_t& handlecublas) {
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int iter = 0;

	float a = ((deltaSalTerm*deltaSalTerm) / (alphaSalTerm *alphaSalTerm)) - (lambda / alphaSalTerm);
	

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
	CalculateB << <numBlocks, numThreads >> > (saliencyControlMap, b, deltaSalTerm, alphaSalTerm, lambda, a, numSuperpixels);
	thrust::device_ptr<float> dev_ptr_pk = thrust::device_pointer_cast(pk);
	
	float *h_energy;
	float *d_energy;
	float *d_max;
	
	int termSize = 3;
	h_energy = (float*)malloc(termSize * sizeof(float));
	checkCudaErrors(cudaMalloc((&d_energy), termSize * sizeof(float)));
	
	
	
	// Starts with 1 because we divide in the gradient 
	float max = 1.0; 
	float energy;	
		
	if (caseAlg > 0) {
		float max_[3] = { 1.0,1.0,1.0 };
	
		checkCudaErrors(cudaMalloc((&d_max), 3 * sizeof(float)));
	
		checkCudaErrors(cudaMemcpy(d_max, max_, 3 * sizeof(float), cudaMemcpyHostToDevice));
		
	}
	else {
		checkCudaErrors(cudaMalloc((&d_max), sizeof(float)));
		
		checkCudaErrors(cudaMemcpy(d_max, &max, sizeof(*d_max), cudaMemcpyHostToDevice));
	}

	CalculateTransIndex << <numBlocks, numThreads >> > (csrValII, csrRow, csrCol, csrVal, pk, knn, numSuperpixels);
	energy = 10;
	float energyPrev = 10, energyDiff = 10;
	int blockReduction;
	
	bool exit = false;
	// http://www.cplusplus.com/reference/cmath/isnan/
	while (!exit && !isnan(energy)) {
	
		switch (caseAlg)
		{
		case 0: // Sal
			NLTVGradient << <numBlocks2, numThreads >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size);
			max = *thrust::max_element(dev_ptr_pk, dev_ptr_pk + size);
			//std::cout << "Iter " << iter << " Max " << max << std::endl;
		
			if (max < 1.0f) max = 1.0f;
			
			DivByMax << <numBlocks2, numThreads >> > (max, size, pk);
			
			NLTVDiv << <numBlocks, numThreads >> > (saliencyControlMap, csrRow, csrCol, csrVal, uk, pk, divP, knn, tau_d, numSuperpixels);
			UpdateVariableNLTVSalTerm << <numBlocks, numThreads >> > (saliencyControlMap, uk, b, divP, tau_p, a, numSuperpixels, d_energy);
			energyNLTVSalTerm << <numBlocks, numThreads >> > (numSuperpixels, uk, saliencyControlMap, alphaSalTerm, deltaSalTerm, lambda, d_energy);
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, 3 * sizeof(*d_energy), cudaMemcpyDeviceToHost));
			//std::cout << "term0 " << term[0] << " term1 " << term[1] << " term2 " << term[2] << std::endl;
			h_energy[1] = -0.5 *(h_energy[1] * (1 / alphaSalTerm));
			h_energy[2] = lambda *h_energy[2];
			energy = alphaSalTerm*h_energy[0] + h_energy[1] + h_energy[2];
		
			break;
		 
		case 1: // SalPlus
			blockReduction = nextPow2(numSuperpixels) / 2;
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSalterm<float, true> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			else
				MaximizationNLTVSalterm<float, false> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			
			if (isPow2(numSuperpixels))
				MinimizationNLTVSalTermEnergy<float, true> << <1, numSuperpixels, numSuperpixels * sizeof(float) >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk, b, tau_p, a, d_energy, d_max, (iter + 1) % 3, blockReduction,deltaSalTerm);
			else
				MinimizationNLTVSalTermEnergy<float, false> << <1, numSuperpixels, numSuperpixels * sizeof(float) >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk, b, tau_p, a, d_energy, d_max, (iter + 1) % 3, blockReduction,deltaSalTerm);

			
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, 3 * sizeof(*d_energy), cudaMemcpyDeviceToHost));
			//std::cout << "term0 " << term[0] << " term1 " << term[1] << " term2 " << term[2] << std::endl;
			h_energy[1] = -0.5 *(h_energy[1] * (1 / alphaSalTerm));
			h_energy[2] = lambda *h_energy[2];
			energy = alphaSalTerm*h_energy[0] + h_energy[1] + h_energy[2];			
			break;

		case 2: // SalStar
			blockReduction = nextPow2(numSuperpixels) / 2;
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSalterm<float, true> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			else
				MaximizationNLTVSalterm<float, false> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
		
			MinimizationNLTVSalTermNoEnergy<float> << <1, numSuperpixels >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk, b, tau_p, a,  d_max, (iter + 1) % 3);
				
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
	printf("IterativeNLTVSaliencyTermGpu Final %f Iter completed %i Final Energy %f\n", timeMS, iter,energyDiff);
	
	checkCudaErrors(cudaFree(d_energy));
	free(h_energy);

	checkCudaErrors(cudaFree(d_max));
	return cudaPeekAtLastError();
}

__host__
cudaError_t NLTVGpu(float* uk, float *b, float *pk, float *divP, const float *saliencyControlMap, const float * csrVal, int *csrValII, const int * csrRow, const int * csrCol, int numSuperpixels,
	const float deltaSalTerm, const float alphaSalTerm, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol, const cublasHandle_t& handlecublas) {
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
	
	thrust::device_ptr<float> dev_ptr_pk = thrust::device_pointer_cast(pk);
	
	float *h_energy;
	float *d_energy;
	float *d_max;

	int termSize = 2;
	h_energy = (float*)malloc(termSize * sizeof(float));
	checkCudaErrors(cudaMalloc((&d_energy), termSize * sizeof(float)));
	

	// Starts with 1 because we divide in the gradient 
	float max = 1.0;
	float divlambda = 1 / lambda;

	float energy;	
	
	if (caseAlg >0) {
		float max_[3] = { 1.0,1.0,1.0 };

		checkCudaErrors(cudaMalloc((&d_max), 3 * sizeof(float)));
		
		checkCudaErrors(cudaMemcpy(d_max, max_, 3 * sizeof(float), cudaMemcpyHostToDevice));
	}
	else {
		checkCudaErrors(cudaMalloc((&d_max), sizeof(float)));
		
		checkCudaErrors(cudaMemcpy(d_max, &max, sizeof(*d_max), cudaMemcpyHostToDevice));
	}

	CalculateTransIndex << <numBlocks, numThreads >> > (csrValII, csrRow, csrCol, csrVal, pk, knn, numSuperpixels);
	energy = 10;
	float energyPrev = 10, energyDiff = 10;
	int blockReduction;

	bool exit = false;
	// http://www.cplusplus.com/reference/cmath/isnan/
	while (!exit && !isnan(energy)) {

		switch (caseAlg)
		{
		case 0: // Sal NLTV
			NLTVGradient << <numBlocks2, numThreads >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size);
			max = *thrust::max_element(dev_ptr_pk, dev_ptr_pk + size);
			//std::cout << "Iter " << iter << " Max " << max << std::endl;

			if (max < 1.0f) max = 1.0f;

			DivByMax << <numBlocks2, numThreads >> > (max, size, pk);

			NLTVDiv << <numBlocks, numThreads >> > (saliencyControlMap, csrRow, csrCol, csrVal, uk, pk, divP, knn, tau_d, numSuperpixels);
			UpdateVariableNLTV << <numBlocks, numThreads >> > (saliencyControlMap, uk, divP, tau_p, divlambda, numSuperpixels, d_energy);
			energyNLTV << <numBlocks, numThreads >> > (numSuperpixels, uk, saliencyControlMap, lambda, d_energy);
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));
			//std::cout << "term0 " << term[0] << " term1 " << term[1] << " term2 " << term[2] << std::endl;
			h_energy[0] = sqrt(h_energy[1]);
			h_energy[1] = 0.5 *lambda *h_energy[1];
			energy = h_energy[0] + h_energy[1] ;

		
			break;
		
		case 1: // SalPlus NLTV
			blockReduction = nextPow2(numSuperpixels) / 2;
			//std::cout << "blockReduction " << blockReduction << " Sp " << numSuperpixels   << std::endl; 
			if (isPow2(numSuperpixels))
				MaximizationNLTVSalterm<float, true> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			else
				MaximizationNLTVSalterm<float, false> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			if (isPow2(numSuperpixels))
				MinimizationNLTVEnergy<float, true> << <1, numSuperpixels, numSuperpixels * sizeof(float) >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk,  tau_p, divlambda, d_energy, d_max, (iter + 1) % 3, blockReduction);
			else
				MinimizationNLTVEnergy<float, false> << <1, numSuperpixels, numSuperpixels * sizeof(float) >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk, tau_p, divlambda, d_energy, d_max, (iter + 1) % 3, blockReduction);
			//sum<float>(numSuperpixels, uk, d_energy);
			checkCudaErrors(cudaMemcpy(h_energy, d_energy, termSize * sizeof(*d_energy), cudaMemcpyDeviceToHost));
			h_energy[0] = sqrt(h_energy[1]);
			h_energy[1] = 0.5 *lambda *h_energy[1];
			energy = h_energy[0] + h_energy[1];

			break;

		case 2: // SalStar NLTV
			blockReduction = nextPow2(numSuperpixels) / 2;			
			if (isPow2(numSuperpixels))
				MaximizationNLTVSalterm<float, true> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);
			else
				MaximizationNLTVSalterm<float, false> << <knn, numSuperpixels, numSuperpixels * sizeof(float) >> > (csrRow, csrCol, csrVal, uk, pk, knn, tau_d, size, d_max, blockReduction, iter % 3, (iter + 1) % 3);

			MinimizationNLTVNoEnergy << <1, numSuperpixels >> > (saliencyControlMap, csrRow, csrCol, csrVal, csrValII, pk, knn, tau_d, numSuperpixels, uk, tau_p, divlambda, d_max, (iter + 1) % 3);
			
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
	
	checkCudaErrors(cudaFree(d_energy));
	free(h_energy);
	
	checkCudaErrors(cudaFree(d_max));
	return cudaPeekAtLastError();
}
