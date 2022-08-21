/*
* Specification of the superpixel parameters for the SLIC.
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-01016-4
* [3] [3] SLIC Superpixels Radhakrishna Achanta, Appu Shaji, Kevin Smith, Aurelien Lucchi, Pascal Fua, and Sabine Susstrunk https://ivrlwww.epfl.ch/supplementary_material/RK_SLICSuperpixels/index.html 
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
#ifndef __SlicParameters_H__
#define __SlicParameters_H__
#pragma once


#include "SuperpixelParameters.h"
class SlicParameters : public SuperpixelParameters
{
private :
	// See [3]
	int compactness;
	// See [3]
	bool do_enforce_connectivity;
public:
	SlicParameters();
	/** SlicParameters constructor
	*  @param sp: number of desired superpixels
	*  @param red: colour of the display
	*  @param green: colour of the display
	*  @param blue: colour of the display
	*  @param compactness: See [3]
	*  @param do_enforce_connectivity: See [3]
	*/
	SlicParameters(int sp, int red, int green, int blue,int compactness, int do_enforce_connectivity);
	SlicParameters& operator =(const SlicParameters& other);
	SlicParameters(const SlicParameters & superpixelParameters);
	~SlicParameters();
	/** GetCompactness
	*/
	int GetCompactness() const;
	/** GetDoEnforceConnectivity
	*/
	int GetDoEnforceConnectivity() const;
};

#endif //__SlicParameters_H__