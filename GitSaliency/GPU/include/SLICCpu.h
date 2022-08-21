// 

/*
* Code to apply the superpixel algorithm SLIC in CPU. The code was taken from [3]
* Some modifications have been coded for a strategy pattern (accomodate more than one superpixel algorithm
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
*
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-0
* [3] SLIC Superpixels Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk https://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html 
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
* Copyright 2020 Eduardo Alcain Ballesteros eduardo.alcain.ballesteros@gmail.com  Ana Muñoz anaisabel.munoz@urjc.es
*/
#ifndef __SLICCpu_H__
#define __SLICCpu_H__
#pragma once

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "ISuperpixel.h"
#include "SlicParameters.h"
#include "StopWatch.h"
#include <string.h>
#include <stdlib.h>     /* malloc, free, rand */

class SLICCpu :public ISuperpixel
{
private:
	// Time to perform the SLIC algorithm
	float timeInMs;
	// Labels of the superpixels in the input image [widthxheight]
	int* clabels;
	// Final number of superpixels
	int  outputNumSuperpixels;
	// SLIC parameters
	SlicParameters slicParameters;
	/** Convert RGB into CIE L*a*b 
	*  @param rin: red channel
	*  @param gin: green channel
	*  @param bin: blue channel
	*  @param sz: size of the image
	*  @param lvec: l channel in CIE L*a*b 
	*  @param avec: a channel in CIE L*a*b 
	*  @param bvec: b channel in CIE L*a*b 
	*/
	template<typename T>
	void rgbtolab(int* rin, int* gin, int* bin, int sz, T* lvec, T* avec, T* bvec);
	/** Set the initial seeds for the SLIC algorithm see [3] for details
	*  @param STEP: 
	*  @param width: 
	*  @param height: 
	*  @param seedIndices: 
	*  @param numseeds: 
	*/
	void getLABXYSeeds(int STEP, int width, int height, int* seedIndices, int* numseeds);
	/** PerformSuperpixelSLIC perform SLIC algorithm see [3] for details
	*  @param lvec: l channel in CIE L*a*b
	*  @param avec: a channel in CIE L*a*b
	*  @param bvec: b channel in CIE L*a*b
	*  @param kseedsl: seeds channel l
	*  @param kseedsa: seeds channel a
	*  @param kseedsb: seeds channel b
	*  @param kseedsx: x coordinate for seeds
	*  @param kseedsy: y coordinate for seeds
	*  @param width: width of the image
	*  @param height: height of the image
	*  @param numseeds:
	*  @param klabels:
	*  @param STEP:
	*  @param compactness:
	*/
	template<typename T>
	void PerformSuperpixelSLIC(T* lvec, T* avec, T* bvec, T* kseedsl, T* kseedsa, T* kseedsb, T* kseedsx, T* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness);
	/** EnforceSuperpixelConnectivity see [3] for details
	*  @param labels: 
	*  @param width: 
	*  @param height:
	*  @param numSuperpixels: s
	*  @param nlabels: 
	*  @param finalNumberOfLabels: 
	*/
	void EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels, int* nlabels, int* finalNumberOfLabels);
public:
	SLICCpu();
	~SLICCpu();
	/** SetSettings in the algorithm
	*  @param superpixelParameters: settings for the SLIC algorithm
	*/
	void SetSettings(const SuperpixelParameters&  superpixelParameters);


	/** Perform the superpixel algorithm
	*  @param img_mat: image to do the oversegmentation
	* return number of final superpixels
	*/
	virtual void SuperPixelSegmentation(const cv::Mat& img);

	/** Get the number of final superpixels
	* return number of final superpixels
	*/
	virtual int GetNumSuperpixels();
	/** Get the matrix [widthxheight] of superpixel labels
	* return superpixel labels
	*/
	virtual int * GetLabels();
	/** Get time of superpixel algorithm
	* return time in ms
	*/
	virtual float GetTimeInMs();
	/** Get algorithm type
	* return type of superpixel algorithm
	*/
	virtual std::string Type();
};

#endif // __SLICCpu_H__