
/*
* Simple code to write matrix into disk and compare later with Octave or Matlab the result
* If you use this software for research purposes, YOU MUST CITE the corresponding
* of the following papers in any resulting publication:
*
* [1] E. Alcaín, A. Muñoz, I. Ramírez, and E. Schiavi. Modelling Sparse Saliency Maps on Manifolds: Numerical Results and Applications, pages 157{175. Springer International Publishing, Cham, 2019.
* [2] Alcaín, E., Muñoz, A.I., Schiavi, E. et al. A non-smooth non-local variational approach to saliency detection in real time. J Real-Time Image Proc (2020). https://doi.org/10.1007/s11554-020-0
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


#ifndef __Matrix_IO_H__
#define __Matrix_IO_H__
#pragma once
#include <iostream>
#include <fstream>

template<typename T>
class MatrixIO
{

public:
	MatrixIO();
	~MatrixIO();
	/** Read the matrix in disk. Format:
	*  rows
	*  cols
	*  data (separated by spaces)
	*  @param file: path of the file channel
	*  @param matrix: structure to store the data (created in the method)
	*/
	void read(std::string& file, T* matrix);
	/** Write the matrix in disk. Format:
	*  rows
	*  cols
	*  data (separated by spaces)
	*  @param fileName: path of the file to
	*  @param rows: rows
	*  @param cols: cols
	*/
	void store(std::string& fileName, const T* vect, int rows, int cols);
	
};


template<typename T>
MatrixIO<T>::MatrixIO() {
}

template<typename T>
MatrixIO<T>::~MatrixIO() {
}


template <typename T>
void MatrixIO<T>::read(std::string& file, T* vect) {
	//std::string line;
	//std::ifstream infile(file);
	//std::string rowsStr, colsStr;
	//std::getline(infile, rowsStr);
	//std::getline(infile, colsStr);
	//int index = 0;
	//int rows = atoi(rowsStr.c_str());
	//int cols = atoi(colsStr.c_str());
	//int size = cols *rows;
	//while (std::getline(infile, line))  // this does the checking!
	//{
	//	std::istringstream iss(line);
	//	T c;

	//	while (iss >> c)
	//	{


	//		if (index < size) {
	//			vect[index] = c;
	//		}
	//		else {
	//			std::cout << "Error" << std::endl;
	//		}
	//		index++;
	//	}
	//}
}


template <typename T>
void MatrixIO<T>::store(std::string& fileName,const T* vect, int rows, int cols) {
	std::ofstream myfile(fileName);
	if (myfile.is_open())
	{
		int size = rows*cols;
		myfile << rows << "\n";
		myfile << cols << "\n";
		for (int count = 0; count < size; count++) {
			myfile << vect[count] << " ";
		}
		myfile.close();
	}
	else std::cout << "Unable to open file";

}


#endif //  __Matrix_IO_H__