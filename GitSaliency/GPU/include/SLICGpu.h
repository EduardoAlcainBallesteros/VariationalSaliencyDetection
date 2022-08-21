#pragma once
/*
* Code to apply the superpixel algorithm SLIC in GPU. The code was taken from [4]. Be aware that the results using this superpixel algorithm
* could be different than the CPU. The implementation is not identical to [3]
* Some modifications have been coded for a strategy pattern (accomodate more than one superpixel algorithm
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
*
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-0
* [3] SLIC Superpixels Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk https://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html
* [4] gSLICr: SLIC superpixels at over 250Hz  Carl Yuheng Ren and Victor Adrian Prisacariu and Ian D Reid  https://www.robots.ox.ac.uk/~victor/gslicr/
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

#ifndef __SLICGpu_H__
#define __SLICGpu_H__

#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "../gSLICr/gSLICr_Lib/gSLICr.h"
#include "ISuperpixel.h"
#include "SlicParameters.h"
#include <string.h>
#include <stdlib.h>     /* malloc, free, rand */
#include "NVTimer.h"
#include "MinMax.h"
#include "helper_cuda.h"

class SLICGpu :public ISuperpixel
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
	/** Convert cv::Mat into the structure gSLICr::UChar4Image
	*  @param inimg: input image
	*  @param outimg: output structure
	*/
	void mat_to_uchar4Image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg);
	/** Convert RGB into CIE L*a*b
	*  @param cols: rows of the image
	*  @param rows: cols of the image
	*  @param settings: settings for the algorithm
	*/
	void settings_SLIC(const int cols, const int rows, gSLICr::objects::settings & settings);
public:
	SLICGpu();
	~SLICGpu();
	/** SetSettings in the algorithm
	*  @param superpixelParameters: settings for the SLIC algorithm
	*/
	void SetSettings(const SuperpixelParameters&  superpixelParameters);

	
	/** Perform the superpixel algorithm 
	*  @param img_mat: image to do the oversegmentation
	* return number of final superpixels
	*/
	virtual void SuperPixelSegmentation(const cv::Mat& img_mat);

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

#endif // __SLICGpu_H__