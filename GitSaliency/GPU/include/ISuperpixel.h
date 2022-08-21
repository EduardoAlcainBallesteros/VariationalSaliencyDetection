/*
* Interface for the superpixel algorithms (strategy pattern)
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

#ifndef __ISuperpixel_H__
#define __ISuperpixel_H__
#pragma once
#include <memory>
#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"

#include "SuperpixelParameters.h"

#include <iostream>

enum class SuperpixelAlgType
{
	SLICGpu,
	SLICCpu
};

class ISuperpixel
{
public:
	/** SetSettings in the algorithm
	*  @param superpixelParameters: settings for the SLIC algorithm
	*/
	virtual void SetSettings(const SuperpixelParameters&  superpixelParameters) = 0;
	/** Perform the superpixel algorithm
	*  @param img_mat: image to do the oversegmentation
	* return number of final superpixels
	*/
	virtual void SuperPixelSegmentation(const cv::Mat& img) = 0;


	virtual int GetNumSuperpixels() = 0;
	/** Get the matrix [widthxheight] of superpixel labels
	* return superpixel labels
	*/
	virtual int * GetLabels() = 0;
	/** Get time of superpixel algorithm
	* return time in ms
	*/
	virtual float GetTimeInMs() = 0;
	/** Get algorithm type
	* return type of superpixel algorithm
	*/
	virtual std::string Type() = 0;

	virtual ~ISuperpixel() = default;
};

using SuperpixelAlgPtr = std::shared_ptr<ISuperpixel>;

/**
* Creates an instance of superpixel based on specified superpixel algorithm
* \param [in] superpixel algorithm
*/
SuperpixelAlgPtr CreateSuperpixelAlg(SuperpixelAlgType superpixelAlgType);

#endif __ISuperpixel_H__