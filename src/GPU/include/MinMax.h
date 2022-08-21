#pragma once
/*
* Interface for implementing min and max functions in the same kernel. The implementation is based on https://github.com/AJcodes/cuda_minmax/blob/master/cuda_minmax/kernel.cu 
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

#ifndef __MIN_MAX_REDUCTION_H__
#define __MIN_MAX_REDUCTION_H__
/** Calculate in a kernel at the same time min and max
*  @param size: size of the vector
*  @param d_max: max value of d_vector
*  @param d_min: min value of d_vector
*  @param d_vector: data
*/
template <class T>
void minmax(int size, T *d_max, T *d_min, const T *d_vector);

/** Calculate max element in the vector
*  @param size: size of the vector
*  @param d_vector: data
*/
template <class T>
T maxElements(int size, const T *d_vector);

#endif

