#pragma once
/*
* Superpixel algorithm is a prestep of the saliency algorithm to pass from pixel domain to superpixel domain. For the algorithms in the following papers
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

#ifndef __SuperpixelParameters_H__
#define __SuperpixelParameters_H__

#include <iostream>
class SuperpixelParameters
{
private :
	// Number of superpixels 
	int sp;
	// Colour for displaying th 
	int red;
	int green;
	int blue;
public:
	SuperpixelParameters();
	SuperpixelParameters(int sp, int red, int green,int blue);
	SuperpixelParameters(const SuperpixelParameters & superpixelParameters);
	~SuperpixelParameters();
	SuperpixelParameters& operator =(const SuperpixelParameters& other);
	int SuperpixelParameters::GetSp() const;
	int SuperpixelParameters::GetRed() const;
	int SuperpixelParameters::GetBlue() const;
	int SuperpixelParameters::GetGreen() const;
	// LNK2001
	// inline virtual const char* WhoIAm();
	inline virtual const char* WhoIAm() {
		return "SuperpixelParameters";

	}
};

#endif // __SuperpixelParameters_H__