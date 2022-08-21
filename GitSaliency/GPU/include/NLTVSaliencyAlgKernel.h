/*
* Kernels use for the implementation of the following saliency models
* E(u) = J_{NLTV,w}(u) + \lambda F(u)				(Saliency NLTV )
* E(u) = J_{NLTV,w}(u) + \lambda F(u) - H(u)		(Saleincy NLTV with SaliencyTerm)
* - J_{NLTV,w} ({\mathbf u})=\sum_{p \in V} \left( \sum_{q \in V,\, pq\in E} w_{pq}|u_q-u_p|^2 \right)^{1/2}
* - F(\mathbf{u})=\frac{1}{\alpha}  ||\mathbf{u}-\mathbf{v}^c ||^2 = \frac{1}{2\alpha} \sum_{p\in V}  |u_p-v^c_p|^2  
* - H(\mathbf{u})=\frac{1}{2\alpha^2}\sum_{p\in V} (1-\delta u_p)^2,
* - \lamdbda The parameter is a positive constant for controlling  the relative importance of the regularizer vs the delity in the functional
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

#pragma once

#ifndef __NLTV_Saliency_Alg_Kernel_H__
#define __NLTV_Saliency_Alg_Kernel_H__



/** Create the matrix of features sp X 5 (Lab + loc (r,c))  
*
*  @param height: height of the image
*  @param width: width of the image
*  @param d_labels: superpixels labels
*  @param d_L: L channel of the image in Lab space
*  @param d_a: a channel of the image in Lab space
*  @param d_b: b channel of the image in Lab space
*  @param sp: number of superpixels
*  @param d_LSp: sum vector for the feature L of each superpixel
*  @param d_aSp: sum vector for the feature a of each superpixel
*  @param d_bSp: sum vector for the feature b of each superpixel
*  @param d_rSp: sum vector for the feature r of each superpixel
*  @param d_cSp: sum vector for the feature c of each superpixel
*  @param numSp: number of elements of each superpixel 
*/
template <typename T>
void CreateSuperpixelAttributeGpu(int height, int width, const int *d_labels, const T *d_L, const T* d_a, const T* d_b, const int sp, T *d_LSp, T  *d_aSp, T *d_bSp, T *d_rSp, T *d_cSp, int *numSp);


/** Calculate Control Map = Contrast Prior (1) x Object Prior (2)
	Calculate Weights adjacency matrix to connect each superpixel through the features colour and locatization (3)
*  (1) CONTRAST PRIOR v_p^c is
*  v_p^{con}=\sum_{q\ne p}\omega_{pq}^{(l)} ||{\bf c}_p-{\bf c}_q||^2,
\quad \omega_{pq}^{(l)}=e^{-\frac{\displaystyle ||{\bf l}_p-{\bf l}_q||^2}{\displaystyle 2\sigma^2}}
*  (2) OBJECT PRIOR v_p^{obj}
*  v_p^{obj}=e^{-\frac{\displaystyle ||{\bf l}_p-\bar{{\bf l}}||^2}{\displaystyle 2\sigma^2}}
*  (3) Weights W needs formula (4)
*  \omega_{pq}=\mbox{exp}\left(  -\frac{||\mathbf{f}_p-\mathbf{f}_q||^2}{2\sigma^2}\right)
*  (4) Vector of characteristics ( colour and locatization) {f}_p=(r \mathbf{c}_p, \mathbf{l}_p) 
*  @param d_saliencyControlMap: Control map calculated by Contrast Prior (1) x Object Prior (2)
*  @param d_weights: Adjacency matrix among superpixels according to (3)
*  @param d_a: vector containing a channel value of each superpixel in the space Lab
*  @param d_b: vector containing b channel value of each superpixel in the space Lab
*  @param d_L: vector containing L channel value of each superpixel in the space Lab
*  @param d_r: vector containing r (row localization in the image grid for each superpixel)
*  @param d_c: vector containing c (col localization in the image grid for each superpixel)
*  @param sp: number of superpixels
*  @param locationPrior: false => Control Map = Contrast Prior (1), true Control Map = Contrast Prior (1) x Object Prior (2)
*  @param r: The r parameter controls the balance between the two features r= 0.9 Formula(4)
*  @param sigma2: sigma2 defines the extent of the non-locality sigma2= 0.05 Formula (4)
*/
template <class T>
void CalculateSaliencyControlMapAndWeightsGpu(T *d_saliencyControlMap, T *d_weights, const  T *d_a, const  T *d_b, const  T *d_L, const  T *d_r, const  T *d_c, int sp, bool locationPrior, const T r, const T sigma2);

/**
* Procedure to reduce the number of total connections (edges of the graph) of each superpixel from sp
* to k-nearest neighboursThe rest of the weights for a superpixel are set to zero. As a result of
* this process, we end up with a sparse weight matrix W^{sp\times k} which is no longer symmetric
*  (1) Weights W needs formula (2)
*  \omega_{pq}=\mbox{exp}\left(  -\frac{||\mathbf{f}_p-\mathbf{f}_q||^2}{2\sigma^2}\right)
*  (2) Vector of characteristics ( colour and locatization) {f}_p=(r \mathbf{c}_p, \mathbf{l}_p)
*  @param d_weights: Adjacency matrix among superpixels according to (1)
*  @param k: k-elements allowed greater than zero for each superpixel the Nh (so far working only for k=5)
*  @param sp: number of superpixels
*/
template <class T>
void CalculateKNNGpu(T *d_weights, int k, int sp);

/**
* Calculate of the minimization of the energy functional
* E(u) = J_{NLTV,w}(u) + \lambda F(u)						(Saliency NLTV ) SalMethod=0
* E(u) = J_{NLTV,w}(u) + \lambda F(u) - H(u)				(Saliency NLTV with SaliencyTerm) Salmethod=1
* tol = 0 iterate until max iter
* Three cases caseAlg:
*	- Sal No optimzed version
* http://asciiflow.com/
                                        energy > tol
         +XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
         |                                                                  X
+--------v-----+ +---------+ +-----------+ +-----------+ +-----------+ +----X----+
|              | |         | |           | |           | |           | |         |
| nltvGradient +->   Max   +-> DivByMax  +->  nltvDiv  +-> UpdateUk  +->  Energy |
|              | |         | |           | |           | |           | ^         |
+--------------+ +---------+ +-----------+ +-----------+ +-----------+ +---------+

*  - SalPlus Optimzed version


                 energy > tol
        XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
        +                                    X
+-------v--------+  +----------------+  +----X-----+
|                |  |                |  |          |
|  Maximization  +-->  Minimization  +-->  Energy  |
|                |  |                |  |          |
+----------------+  +----------------+  +----------+

*	- SalStar optimzed version without energy criteria


            iter < maxIter
        XXXXXXXXXXXXXXXXXXXXXXX
        +                     X
+-------v--------+  +---------X------+
|                |  |                |
|  Maximization  +-->  Minimization  +
|                |  |                |
+----------------+  +----------------+
*/

template <class T>
void NLTVSaliency(T* d_uk, T *d_bPD, T *d_pk, T *d_divP, const T *d_saliencyControlMap, const T * d_values, int *d_trans, const int * d_rows, const int * d_cols,const int numSuperpixels,
	const float delta, const float alpha, const float lambda, const float tau_d, const float tau_p, const int knn, const int maxIter, const int caseAlg, const float tol, const int salMethod, const cublasHandle_t& handlecublas);


/** Convert image RGB into Lab
*  https://en.wikipedia.org/wiki/CIELAB_color_space
*  @param d_rin: red channel of the image
*  @param d_gin: green channel of the image
*  @param d_bin: blue channel of the image
*  @param sz: number of elements in the image width x height
*  @param d_L: L channel of the image in Lab space
*  @param d_a: a channel of the image in Lab space
*  @param d_b: b channel of the image in Lab space
*/
template <class T>
void RgbToLabGpu(const int* d_rin, const int* d_gin, const int* d_bin, const int sz, T* d_L, T* d_a, T* d_b);





#endif //__NLTV_Saliency_Alg_Kernel_H__


