
#include "../include/NLTVSaliencyAlg.h"





#if MATLAB_SUPPORT
#pragma message("Compiled in _X64")
#include "MatrixMatFile.h"
#endif



NLTVSaliencyAlg::NLTVSaliencyAlg()
{	
	cusparseCreate(&handle);
	cublasCreate(&handlecublas);

	
	cusparseCreateMatDescr(&descrA);
	cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
}


NLTVSaliencyAlg::~NLTVSaliencyAlg()
{
	cublasDestroy(handlecublas);
	cusparseDestroy(handle);
	cusparseDestroyMatDescr(descrA);
}


float NLTVSaliencyAlg::Execute(const std::string& imagePath, const SlicParameters & superpixelsParameters, const SaliencyParameters & parameters,float *slicTimeInMs)
{
	float saliencyTimeInMs;

	Mat oldFrame, frame;

	this->superpixelsParameters = superpixelsParameters;
	this->saliencyParameters = parameters;
	SuperpixelAlgPtr superpixelAlgPtr = CreateSuperpixelAlg(this->saliencyParameters.spMethod);
	
	StopWatchInterface *saliencyTimer; 

	
	sdkCreateTimer(&saliencyTimer);
	
	frame = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
	std::cout << "Imgname " << imagePath << " Segs " << this->superpixelsParameters.GetSp();
	std::cout << "[" << frame.cols << "," << frame.rows << "]" << std::endl;
	superpixelAlgPtr->SetSettings(this->superpixelsParameters);
	superpixelAlgPtr->SuperPixelSegmentation(frame);
	int * h_labels = superpixelAlgPtr->GetLabels();


	

	Size s(frame.cols, frame.rows);

	Mat boundry_draw_frame; boundry_draw_frame.create(s, CV_8UC3);

	int key;
	int height = frame.rows;
	int width = frame.cols;
	int size = height *width;
	int numSuperpixels;
	
	int *d_labels=NULL;
	float *d_a = NULL;
	float *d_b = NULL;
	float * d_L = NULL;

	float *d_aSp = NULL;
	float *d_bSp = NULL;
	float * d_LSp = NULL;
	float * d_rSp = NULL;
	float * d_cSp = NULL;
	float * d_saliencyControlMap = NULL;
	
	float *d_weights = NULL;

	int * d_numSp = NULL;
	int *d_rin=NULL, *d_gin=NULL, *d_bin=NULL;
		
	
	sdkResetTimer(&saliencyTimer); sdkStartTimer(&saliencyTimer);

	float *h_to_store = NULL;
	float *h_uk = NULL;
		int *h_rin = (int*)malloc(sizeof(int)      * size);
		int *h_gin = (int*)malloc(sizeof(int)      * size);
		int *h_bin = (int*)malloc(sizeof(int)      * size);
	
		const cv::Vec3b *imgColour = frame.ptr<cv::Vec3b>(0);
		for (int i = 0; i < size; i++)
		{
			uchar blue = imgColour[i].val[0];
			uchar green = imgColour[i].val[1];
			uchar red = imgColour[i].val[2];
			h_rin[i] = red;
			h_gin[i] = green;
			h_bin[i] = blue;

		}

		checkCudaErrors(cudaMalloc((&d_labels), size * sizeof(int)));
		checkCudaErrors(cudaMalloc((&d_bin), size * sizeof(int)));		
		checkCudaErrors(cudaMalloc((&d_gin), size * sizeof(int)));
		checkCudaErrors(cudaMalloc((&d_rin), size * sizeof(int)));
		
		checkCudaErrors(cudaMalloc((&d_L), size * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_a), size * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_b), size * sizeof(float)));
		checkCudaErrors(cudaMemcpy(d_labels, h_labels, size * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_bin, h_bin, size * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_rin, h_rin, size * sizeof(int), cudaMemcpyHostToDevice));
		checkCudaErrors(cudaMemcpy(d_gin, h_gin, size * sizeof(int), cudaMemcpyHostToDevice));


		RgbToLabGpu(d_rin, d_gin, d_bin, size, d_L, d_a, d_b);

#ifdef DEBUG_MATRIX	 
		//MatrixIO<float> matrixIO;
		//MatrixIO<int> matrixIOLabels;
		MatrixMatFile matFile;


		float * h_L, *h_a, *h_b;
		
		h_L = (float*)malloc(sizeof(float)* height * width);
		h_a = (float*)malloc(sizeof(float)* height * width);
		h_b = (float*)malloc(sizeof(float)* height * width);
	

		checkCudaErrors(cudaMemcpy(h_L, d_L, height * width * sizeof(*d_L), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_a, d_a, height * width * sizeof(*d_a), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_b, d_b, height * width * sizeof(*d_b), cudaMemcpyDeviceToHost));
		

		matFile.WriteOutputMatFile("lvec_GPU.mat", "lvec_GPU", h_L, height, width);
		matFile.WriteOutputMatFile("avec_GPU.mat", "avec_GPU", h_a, height, width);
		matFile.WriteOutputMatFile("bvec_GPU.mat", "bvec_GPU", h_b, height, width);


		free(h_L);
		free(h_a);
		free(h_b);
		

#endif


		numSuperpixels = superpixelAlgPtr->GetNumSuperpixels();
		
		checkCudaErrors(cudaMalloc((&d_LSp), numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_aSp), numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_bSp), numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_rSp), numSuperpixels * sizeof(float)));

		checkCudaErrors(cudaMalloc((&d_cSp), numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMalloc((&d_numSp), numSuperpixels * sizeof(int)));
		checkCudaErrors(cudaMalloc((&d_saliencyControlMap), numSuperpixels * sizeof(int)));
		checkCudaErrors(cudaMalloc((&d_weights), numSuperpixels* numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_LSp, 0, numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_aSp, 0, numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_bSp, 0, numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_rSp, 0, numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_cSp, 0, numSuperpixels * sizeof(float)));
		checkCudaErrors(cudaMemset(d_numSp, 0, numSuperpixels * sizeof(int)));
	
		CreateSuperpixelAttributeGpu(height, width, d_labels, d_L, d_a, d_b, numSuperpixels, d_LSp, d_aSp, d_bSp, d_rSp, d_cSp, d_numSp);

#ifdef DEBUG_MATRIX	 
		

		
		float * h_vec_L, *h_vec_a, *h_vec_b, *h_vec_r, *h_vec_c, *hf_labels;
		int *hi_labels;
		h_vec_L = (float*)malloc(sizeof(float)* numSuperpixels);
		h_vec_a = (float*)malloc(sizeof(float)* numSuperpixels);
		h_vec_b = (float*)malloc(sizeof(float)* numSuperpixels);
		h_vec_r = (float*)malloc(sizeof(float)* numSuperpixels);
		h_vec_c = (float*)malloc(sizeof(float)* numSuperpixels);
		hf_labels = (float*)malloc(sizeof(float)* numSuperpixels * numSuperpixels);
		hi_labels = (int*)malloc(sizeof(int)* numSuperpixels * numSuperpixels);

		checkCudaErrors(cudaMemcpy(h_vec_L, d_LSp, numSuperpixels * sizeof(*d_LSp), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_vec_a, d_aSp, numSuperpixels * sizeof(*d_aSp), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_vec_b, d_bSp, numSuperpixels * sizeof(*d_bSp), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_vec_r, d_rSp, numSuperpixels * sizeof(*d_rSp), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_vec_c, d_cSp, numSuperpixels * sizeof(*d_cSp), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(hi_labels, d_labels, numSuperpixels* numSuperpixels * sizeof(*d_labels), cudaMemcpyDeviceToHost));
		/*matrixIO.store("L_GPU.txt", h_vec_L, 1, numSuperpixels);
		matrixIO.store("a_GPU.txt", h_vec_a, 1, numSuperpixels);
		matrixIO.store("b_GPU.txt", h_vec_b, 1, numSuperpixels);
		matrixIO.store("r_GPU.txt", h_vec_r, 1, numSuperpixels);
		matrixIO.store("c_GPU.txt", h_vec_c, 1, numSuperpixels);*/

		matFile.WriteOutputMatFile("L_GPU.mat", "L_GPU", h_vec_L, 1, numSuperpixels);
		matFile.WriteOutputMatFile("a_GPU.mat", "a_GPU", h_vec_a, 1, numSuperpixels);
		matFile.WriteOutputMatFile("b_GPU.mat", "b_GPU", h_vec_b, 1, numSuperpixels);
		matFile.WriteOutputMatFile("r_GPU.mat", "r_GPU", h_vec_r, 1, numSuperpixels);
		matFile.WriteOutputMatFile("c_GPU.mat", "c_GPU", h_vec_c, 1, numSuperpixels);
	
		for (int it = 0; it < numSuperpixels*numSuperpixels; it++) {
		//	std::cout << "Llega asd " << it  << std::endl;
			hf_labels[it] = hi_labels[it];
		}
		
		matFile.WriteOutputMatFile("labels_GPU.mat", "labels_GPU", hf_labels, numSuperpixels, numSuperpixels);
		
		free(h_vec_L);
		free(h_vec_a);
		free(h_vec_b);
		free(h_vec_r);
		free(h_vec_c);

		
		free(hf_labels);
		free(hi_labels);
	
#endif
		
		CalculateSaliencyControlMapAndWeightsGpu(d_saliencyControlMap, d_weights, d_aSp, d_bSp, d_LSp, d_rSp, d_cSp, numSuperpixels, parameters.locationPrior, parameters.r, parameters.sigma2);
		
#ifdef DEBUG_MATRIX	 
		float * h_weights;
		h_to_store = (float*)malloc(sizeof(float)  * numSuperpixels);
		h_weights = (float*)malloc(sizeof(float) *numSuperpixels  * numSuperpixels);
		checkCudaErrors(cudaMemcpy(h_to_store, d_saliencyControlMap, numSuperpixels * sizeof(*d_saliencyControlMap), cudaMemcpyDeviceToHost));
		checkCudaErrors(cudaMemcpy(h_weights, d_weights, numSuperpixels* numSuperpixels * sizeof(*d_saliencyControlMap), cudaMemcpyDeviceToHost));

		/*matrixIO.store("controlMapGPU.txt",h_to_store, 1, numSuperpixels);
		matrixIO.store("weightsGPU.txt", h_weights, numSuperpixels, numSuperpixels);*/
		
		matFile.WriteOutputMatFile("controlMap_GPU.mat", "controlMap_GPU", h_to_store, 1, numSuperpixels);
		matFile.WriteOutputMatFile("weights_GPU.mat", "weights_GPU", h_weights, numSuperpixels, numSuperpixels);
		free(h_to_store);
		free(h_weights);
	
#endif
		CalculateKNNGpu(d_weights, parameters.k, numSuperpixels);
#ifdef DEBUG_MATRIX	
		
		h_weights = (float*)malloc(sizeof(float) *numSuperpixels  * numSuperpixels);
		checkCudaErrors(cudaMemcpy(h_weights, d_weights, numSuperpixels* numSuperpixels * sizeof(*d_saliencyControlMap), cudaMemcpyDeviceToHost));
		matFile.WriteOutputMatFile("weightsKnn_GPU.mat", "weightsKnn_GPU", h_weights, numSuperpixels, numSuperpixels);
		free(h_weights);
#endif
	
		///////////////
		//-- Initialize cuSPARSE

		float alpha = 1.;
		float beta = 0.;

		float *d_wknn_denseTrans=NULL,  *d_uk = NULL, *d_pk = NULL, *d_divP = NULL,  *d_bPD = NULL;;

		checkCudaErrors(cudaMalloc(&d_uk, numSuperpixels * sizeof(*d_uk)));
		checkCudaErrors(cudaMalloc(&d_divP, numSuperpixels * sizeof(*d_divP)));
		checkCudaErrors(cudaMalloc(&d_bPD, numSuperpixels * sizeof(*d_b)));
		checkCudaErrors(cudaMalloc(&d_wknn_denseTrans, numSuperpixels * numSuperpixels * sizeof(*d_wknn_denseTrans)));
		checkCudaErrors(cudaMemcpy(d_uk, d_saliencyControlMap, numSuperpixels * sizeof(*d_saliencyControlMap), cudaMemcpyDeviceToDevice));
		

		
		// cublas works with column major we need to transpose the matrix
		cublasSgeam(handlecublas, CUBLAS_OP_T, CUBLAS_OP_N, numSuperpixels, numSuperpixels, &alpha, d_weights, numSuperpixels, &beta, d_weights, numSuperpixels, d_wknn_denseTrans, numSuperpixels);

		

		// --- Number of nonzero elements in dense matrix
		int nnz = 0;           
		// --- Leading dimension of dense matrix
		const int lda = numSuperpixels;                      
		// --- Device side number of nonzero elements per row								
		int *d_nnzPerVector;

		checkCudaErrors(cudaMalloc(&d_nnzPerVector, numSuperpixels * sizeof(*d_nnzPerVector)));
	
		
		cusparseSnnz(handle, CUSPARSE_DIRECTION_ROW, numSuperpixels, numSuperpixels, descrA, d_wknn_denseTrans, lda, d_nnzPerVector, &nnz);
		checkCudaErrors(cudaMalloc(&d_pk, parameters.k * numSuperpixels * sizeof(*d_pk)));
		
		checkCudaErrors(cudaMemset(d_pk, 0, sizeof(*d_pk)*parameters.k  * numSuperpixels));
		

		// --- Device side dense matrix
		float *d_values;
		int *d_rows;
		int *d_cols;
		checkCudaErrors(cudaMalloc(&d_values, nnz * sizeof(*d_values)));
		
		

		checkCudaErrors(cudaMalloc(&d_rows, (numSuperpixels + 1) * sizeof(*d_rows)));
		checkCudaErrors(cudaMalloc(&d_cols, nnz * sizeof(*d_cols)));
		
		// weights to CSR Format
		cusparseSdense2csr(handle, numSuperpixels, numSuperpixels, descrA, d_wknn_denseTrans, lda, d_nnzPerVector, d_values, d_rows, d_cols);
		int *h_trans = (int*)malloc(sizeof(int)*parameters.k * numSuperpixels);
		for (int i = 0; i < parameters.k * numSuperpixels; i++) {
			h_trans[i] = -1;
		}
		// lookup table (LUT) with the transposed information (Trans NNZ). This LUT
		// vector stores the transposed result for each index in the compact matrix W^{ sp\times k } where "-1" entry means the
		// transposed value is zero and a value [0; k) indicates the index of the column where the value in the compact
		// matrix is located
		int * d_trans;
		checkCudaErrors(cudaMalloc(&d_trans, parameters.k * numSuperpixels * sizeof(*d_trans)));
		checkCudaErrors(cudaMemcpy(d_trans, h_trans, parameters.k * numSuperpixels * sizeof(*d_trans), cudaMemcpyHostToDevice));
		NLTVSaliency(d_uk, d_b, d_pk, d_divP, d_saliencyControlMap, d_values, d_trans, d_rows, d_cols, numSuperpixels, parameters.deltaSalTerm, parameters.alphaSalTerm, parameters.lambda, parameters.tau_d, parameters.tau_p, parameters.k, parameters.maxIter, parameters.algCudaVariant, parameters.tol, parameters.salMethod, handlecublas);
	
		h_uk = (float*)malloc(sizeof(float)*numSuperpixels);
		checkCudaErrors(cudaMemcpy(h_uk, d_uk, numSuperpixels * sizeof(*d_uk), cudaMemcpyDeviceToHost));
#ifdef DEBUG_MATRIX	 	
		matFile.WriteOutputMatFile("finalUk_GPU.mat", "finalUk_GPU", h_uk, 1, numSuperpixels);
#endif
		Mat saliencyMap(frame.rows, frame.cols, CV_8U, Scalar(0, 0, 0));
		DrawMapSaliencyResult(saliencyMap, h_uk, h_labels,numSuperpixels );
		

		DrawSuperpixelGrid(frame, boundry_draw_frame, cv::Scalar(superpixelsParameters.GetBlue(),superpixelsParameters.GetGreen(), superpixelsParameters.GetRed()), h_labels);
		
		sdkStopTimer(&saliencyTimer);
		saliencyTimeInMs = sdkGetTimerValue(&saliencyTimer);
		*slicTimeInMs = superpixelAlgPtr->GetTimeInMs();
		std::cout << "Real superpixels " << numSuperpixels << " Time Sal in ms " << saliencyTimeInMs << std::endl;
		if (parameters.imgDisplayOn) {
			
			imshow("segmentation", boundry_draw_frame);
			cv::imshow("Saliency", saliencyMap);
			waitKey();
		}
		if (parameters.imgWrite) {

			cv::imwrite("output.png", saliencyMap);
		}
	

		/* Free resources CUDA */
		
		checkCudaErrors(cudaFree(d_labels));
		checkCudaErrors(cudaFree(d_rin));
		checkCudaErrors(cudaFree(d_gin));
		checkCudaErrors(cudaFree(d_bin));



		checkCudaErrors(cudaFree(d_a));
		checkCudaErrors(cudaFree(d_b));
		checkCudaErrors(cudaFree(d_L));

		checkCudaErrors(cudaFree(d_LSp));
		checkCudaErrors(cudaFree(d_aSp));
		checkCudaErrors(cudaFree(d_bSp));
		checkCudaErrors(cudaFree(d_rSp));
		checkCudaErrors(cudaFree(d_cSp));
		checkCudaErrors(cudaFree(d_numSp));

		checkCudaErrors(cudaFree(d_saliencyControlMap));


		checkCudaErrors(cudaFree(d_weights));

		checkCudaErrors(cudaFree(d_wknn_denseTrans));
		checkCudaErrors(cudaFree(d_nnzPerVector));
		checkCudaErrors(cudaFree(d_trans));

		checkCudaErrors(cudaFree(d_rows));
		checkCudaErrors(cudaFree(d_cols));
		checkCudaErrors(cudaFree(d_values));
		checkCudaErrors(cudaFree(d_bPD));
		checkCudaErrors(cudaFree(d_uk));
		checkCudaErrors(cudaFree(d_pk));
		checkCudaErrors(cudaFree(d_divP));
	

		/* Free resources CUDA ENDS */


		/* Free resources heap */

		free(h_rin);
		h_rin = NULL;
		free(h_gin);
		h_gin = NULL;
		free(h_bin);
		h_bin = NULL;

		free(h_uk);
		h_uk = NULL;

		/* Free resources heap ENDS */
		



	sdkDeleteTimer(&saliencyTimer);

	return saliencyTimeInMs;

}


void NLTVSaliencyAlg::DrawMapSaliencyResult(cv::Mat& saliency_mat, float * saliency_map_sp, const  int * sp_labels,const  int outputNumSuperpixels) {

	// Min and max for normalization
	float minValue = *std::min_element((float*)saliency_map_sp, (float*)saliency_map_sp + outputNumSuperpixels);
	float maxValue = *std::max_element((float*)saliency_map_sp, (float*)saliency_map_sp + outputNumSuperpixels);
	float range = maxValue - minValue;
	
	int aux;
	float normValue;
	uchar * segmPtr = saliency_mat.ptr<uchar>(0);
	// We normalize the range of the solution u in the superpixel domain and then, we
	// project back onto the image domain to obtain the final output u (saliency_map_sp)
	for (int i = 0; i < outputNumSuperpixels; i++) {
		saliency_map_sp[i] = 255.0f * ((saliency_map_sp[i] - minValue) / range);	
	}

	for (int k = 0; k < saliency_mat.rows; k++) {
		aux = k*saliency_mat.cols;
		for (int j = 0; j < saliency_mat.cols; j++) {
			segmPtr[aux + j] = (uchar)saliency_map_sp[sp_labels[aux + j]];
		}
	}
	
}

void NLTVSaliencyAlg::DrawSuperpixelGrid(const cv::Mat& input_mat, cv::Mat& sp_grid_mat, const cv::Scalar& colour, const int * sp_labels) {
	const cv::Vec3b *img = input_mat.ptr<cv::Vec3b>(0);
	cv::Vec3b *imgSeg = sp_grid_mat.ptr<cv::Vec3b>(0);
	int aux;
	
	for (int i = 0; i < input_mat.rows; i++) {
		aux = i*input_mat.cols;
		for (int j = 0; j< input_mat.cols; j++) {
			if (i == 0 || j == 0 || i > input_mat.rows - 2 || j > input_mat.cols - 2) {
				uchar blue = img[aux + j].val[0];
				uchar green = img[aux + j].val[1];
				uchar red = img[aux + j].val[2];
				imgSeg[aux + j].val[0] = blue;
				imgSeg[aux + j].val[1] = green;
				imgSeg[aux + j].val[2] = red;
			}
			else {
				if (sp_labels[aux + j] != sp_labels[aux + j + 1] ||
					sp_labels[aux + j] != sp_labels[aux + j - 1] ||
					sp_labels[aux + j] != sp_labels[(i - 1)*input_mat.cols + j] ||
					sp_labels[aux + j] != sp_labels[(i + 1)*input_mat.cols + j]) {
					imgSeg[aux + j].val[0] = colour.val[0];
					imgSeg[aux + j].val[1] = colour.val[1];
					imgSeg[aux + j].val[2] = colour.val[2];
				}
				else {
					uchar blue = img[aux + j].val[0];
					uchar green = img[aux + j].val[1];
					uchar red = img[aux + j].val[2];
					imgSeg[aux + j].val[0] = blue;
					imgSeg[aux + j].val[1] = green;
					imgSeg[aux + j].val[2] = red;
				}
			}
		}
	}

}