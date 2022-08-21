#include <iostream>
#include "../include/NLTVSaliencyAlg.h"
#include <filesystem>

#if BOOST_SUPPORT
#pragma message("Compiled Boost")
#include <boost\filesystem.hpp>
#include <boost/algorithm/string.hpp>
#include<boost/tokenizer.hpp>
#include "boost/format.hpp"
using namespace boost;

using namespace boost::filesystem;

using namespace std;
using namespace cv;

#endif

// Variables
int red, blue, green;
int superPixels, compactness,  locationPrior, spMethod, salMethod, spixel_size;
int  algCudaVariant, cudaDevice,avg,many;

int knn, execMethod, maxIter;
float r,sigma2,lambda, tau_p, tau_d, tol, delta, alpha;

bool displayFlag, writeFlag, singleFlag, boundariesFlag, locationPriorFlag, priorMapFlag, displayEnergyDebugFlag, displayImgDebugFlag;
cv::String imagePath,  pathOut, imgExtension, pathImages, videoPath;

//#define ITW 
const String keys =
"{help h usage ? |      | print this message   }"
#ifdef ITW
"{imgName        | C:\\Edu\\Personal\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT\\Imgs\\280.jpg    | image to apply saliency algorithm    }"
//"{imgName        | C:\\Edu\\dyve\\Eduardo\\Knoop\\KnoopAll\\15.png    | image to apply saliency algorithm    }"
"{videoPath    |D:\\Users\\ealcain\\PhD_Edu\\DB_Images\\DAVIS\\JPEGImages\\480p\\rallye.avi   | path of the video    }"
"{pathImages   |  C:\\Edu\\Personal\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT\\Imgs\\   | path for images      }"
#else
//"{imgName        | D:\\Users\\ealcain\\PhD_Edu\\DB_Images\\2_2721.jpg    | image to apply saliency algorithm    }"
"{imgName | D:\\Users\\ealcain\\PhD_Edu\\DB_Images\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT\\Imgs\\100376.jpg | image to apply saliency algorithm     }"
"{videoPath    |D:\\Users\\ealcain\\PhD_Edu\\DB_Images\\DAVIS\\JPEGImages\\480p\\bear.avi   | path of the video   }"
"{pathImages   |  D:\\Users\\ealcain\\PhD_Edu\\DB_Images\\MSRA10K_Imgs_GT\\MSRA10K_Imgs_GT\\Imgs\\   | path for images      }"
#endif

"{superPixels    |300   | number of superpixels   }"
"{compactness    |10     | compactness in SLIC               }"
"{red           |0    | red colour component for grid         }"
"{blue            | 0 | blue colour component for grid }"
"{green        |0   | green colour component for grid     }"
"{locationPrior   |   1   | location activated or not       }"
"{spMethod   |0 | Superpixel method: SLICGpu=0,SLICCpu=1         }"
"{salMethod   |1 | NLTV=0 NLTVSalTerm=1 }"
"{execMethod   |1   | 0  a folder 1 one image  2 video     }"
"{knn   |  5   | k-nearest neighbours      }"
"{display   | 1   | display images  after computation }"
"{write   | 0   | write images after computation  }"
"{imgExtension   |  .jpg   | extension used in folder mode see -execMethod=0      }"
"{pathOut   |  .\\   | path for results      }"
"{r   |  0.9  | The r parameter controls the balance between the two features: colour and location }"
"{sigma2   |  0.1  | \sigma2 defines the extent of the non-locality }"
"{lambda   |  1  | a positive constant for controlling the relative importance of the regularizer vs the fidelity in the functional }"
"{tau_p   |  0.3   | descent primal discretization time (Primal-Dual Algorithm)}"
"{tau_d   |  0.33   | an ascent dual discretization time (Primal-Dual Algorithm)      }"
"{tol   |   0.001  | epsilon  a stopping criteria, difference between consecutive values of the energy functional,tol=0  the algorithm iterate until maxIter}"
"{maxIter   | 100   | number maximum of iterations in the Primal-Dual Algorithm 0 means stop when the convergence criteria has reached    }"
"{delta  | 0.2  |  delta > 0 acts as a threshold in the region domain separating background and saliency }"
"{alpha   |  1.5   | models the relative importance of the likelihood and the saliency term }"
"{displayEnergyDebug   | 0   | display energy during the computation }"
"{displayImgDebug   | 0   | display images of the saliency during computation }"
"{algCudaVariant   | 1   | optimization to calculate the saliency algorithm 0=Sal 1=Sal+ 1=Sal*}"
"{cudaDevice   | 0   | cuda device 0 Tesla K40c 1 GTX Titan Black }"
"{avg   | 0   | get the avg for the all executions 0 only one execution 1 executes as many time as the parameter @many }"
"{many   | 10   | number of times to calculate the saliency algorithm }"
;


#if BOOST_SUPPORT

void testReadFolder()
{
	//StopWatch stopWatch;
	/*std::ofstream myfile;
	myfile.open("tt.txt");*/

	//stopWatch.Start();
	// list all files in current directory.
	//You could put any file path in here, e.g. "/home/me/mwah" to list that directory
	path p(pathImages.c_str());
	std::string pattern = "_edge";
	std::string pathOut(pathOut);
	directory_iterator end_itr;
	printf("Test folder %s \n extension %s\n", pathImages.c_str(), imgExtension);
	int count = 0;

	NLTVSaliencyAlg nltvSalAlg;
	NLTVSaliencyAlg::SaliencyParameters saliencyParameters;
	int k = 5;

	cudaDeviceProp deviceProp;
	cudaSetDevice(cudaDevice);
	cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, cudaDevice);

	if (error_id == cudaSuccess)
	{
		printf(" Device %d: %s\n", cudaDevice, deviceProp.name);
	}

	saliencyParameters.lambda = 0.1f;
	saliencyParameters.tau_p = 0.3f;
	saliencyParameters.tau_d = 0.33f;
	saliencyParameters.tol = 0.00001f;
	saliencyParameters.maxIter = maxIter;
	saliencyParameters.deltaSalTerm = 1.1;
	saliencyParameters.alphaSalTerm = 1.5;
	saliencyParameters.energyDisplayOn = false;
	saliencyParameters.k = k;
	saliencyParameters.choice = caseAlgCuda;
	saliencyParameters.energyType = energyType;
	saliencyParameters.imgDisplayOn = displayFlag;
	saliencyParameters.tol = tol;
	

	
	float totalInMs, totalSlicTimeinMs = 0.0f, totalSaliencyTimeinMs = 0.0f;
	float slicTimeinMs, saliencyTimeinMs;
	// cycle through the directory
	//&& count >9900
	for (directory_iterator itr(p); itr != end_itr; ++itr)
	{

		//	// If it's not a directory, list it. If you want to list directories too, just remove this check.
		if (is_regular_file(itr->path())) {
			// assign current file name to current_file and echo it out to the console.
			std::string path = itr->path().string();
			std::string current_file = itr->path().stem().string();
			std::string parentPath = itr->path().parent_path().string();

			std::stringstream edgeMapImg;

			//edgeMapImg << boost::format("%s%s%s.png") % pathMap % current_file % pattern;
			// http://www.cplusplus.com/reference/string/string/compare/
			if (itr->path().extension().compare(imgExtension) == 0) {

				cv::Mat imgColour = cv::imread(path, CV_LOAD_IMAGE_COLOR);
				std::cout <<"#:" << count << "  "<< current_file << std::endl;
				//myfile << current_file << "\n";
				
				//testSaliencyFindParameters(imgColour, path, edgeMapImg.str());
				saliencyTimeinMs = nltvSalAlg.Execute(path, superPixels, compactness, saliencyParameters, &slicTimeinMs);
				//cout << "\rSaliency in:[" << totalInMs << "]ms" << flush;
				/*if (count == 0)
					break;
				*/
				totalSlicTimeinMs += slicTimeinMs;
				totalSaliencyTimeinMs += saliencyTimeinMs;
				count++;
			}
			

		}



	}
	totalInMs = totalSlicTimeinMs + totalSaliencyTimeinMs;	
	std::cout << "Total in Ms " << totalInMs << " each " << totalInMs / (float)count << " many " << count << std::endl;
	std::cout << "SLIC in Ms " << totalSlicTimeinMs << " each " << totalSlicTimeinMs / (float)count << " many " << count << std::endl;
	std::cout << "Saliency in Ms " << totalSaliencyTimeinMs << " each " << totalSaliencyTimeinMs / (float)count << " many " << count << std::endl;
	//stopWatch.Stop();
	//cout << "\rSaliency in:[" << totalInMs/1000.0f << "]ms many " << count << std::endl;
}

#endif


void print_parameters(const NLTVSaliencyAlg::SaliencyParameters& saliencyParameters) {
	std::cout << "### Saliency Parameters ###" << std::endl;
	string salMethodStr;
	string algCudaVariantStr;
	string spMethodStr;
	switch (saliencyParameters.salMethod)
	{
	case NLTVSaliencyAlg::SalMethod::SalNLTV:
		salMethodStr = "SalNLTV";
		break;
	case NLTVSaliencyAlg::SalMethod::SalNLTVSalTerm:
		salMethodStr = "SalNLTVSalTerm"; 
		break;
	
	}
	switch (saliencyParameters.algCudaVariant)
	{
	case NLTVSaliencyAlg::AlgCudaVariant::Sal:
		algCudaVariantStr ="Sal";
		break;
	case NLTVSaliencyAlg::AlgCudaVariant::SalPlus:
		algCudaVariantStr = "Sal+";
		break;
	case NLTVSaliencyAlg::AlgCudaVariant::SalStar:
		algCudaVariantStr = "Sal*";
		break;
	}
	switch (saliencyParameters.spMethod)
	{
	case SuperpixelAlgType::SLICGpu:
		spMethodStr ="SLICGpu";
		break;
	case SuperpixelAlgType::SLICCpu:
		spMethodStr = "SLICCpu";		
		break;
	default:
		break;
	}
	std::cout << "[" << salMethodStr << "," << algCudaVariantStr << "," << spMethodStr << "]" << endl;
	std::cout << "Tau_d [" << saliencyParameters.tau_d << "]";
	std::cout << " Tau_p [" << saliencyParameters.tau_p << "]";
	std::cout << " Lambda [" << saliencyParameters.lambda << "]" << std::endl;;
	std::cout << "Alpha [" << saliencyParameters.alphaSalTerm << "]";
	std::cout << " Delta [" << saliencyParameters.deltaSalTerm << "]";	
	std::cout << " Tol [" << saliencyParameters.tol << "]" << std::endl;
	std::cout << "K [" << saliencyParameters.k << "]";
	std::cout << " MaxIter [" << saliencyParameters.maxIter << "]" << std::endl;
	std::cout << "EnergyDisplay [" << saliencyParameters.energyDisplayOn << "]";
	
	std::cout << " Display [" << saliencyParameters.imgDisplayOn << "]";
	std::cout << " Write [" << saliencyParameters.imgWrite << "]" << std::endl;
	
	
	

	std::cout << "#########################" << std::endl;
}
// NLTVSalTerm Sal No optimzed version
// NLTVSaliencyCuda.exe -lambda=1 -tau_p=0.3 -tau_d=0.33  -delta=0.2 -alpha=1.5 -algCudaVariant=0 -salMethod=1 -spMethod=1
// NLTVSalTerm Sal optimzed version
// NLTVSaliencyCuda.exe -lambda=1 -tau_p=0.3 -tau_d=0.33  -delta=0.2 -alpha=1.5 -algCudaVariant=1 -salMethod=1 -spMethod=1
// NLTVSalTerm Sal optimzed version no energy
// NLTVSaliencyCuda.exe -lambda=1 -tau_p=0.3 -tau_d=0.33  -delta=0.2 -alpha=1.5 -algCudaVariant=2 -salMethod=1 -spMethod=1 -tol=0

// NLTV Sal No optimzed version
// NLTVSaliencyCuda.exe -lambda=0.1 -tau_p=0.3 -tau_d=0.03  -algCudaVariant=0 -salMethod=0 -spMethod=1
// NLTV Sal optimzed version
// NLTVSaliencyCuda.exe -lambda=0.1 -tau_p=0.3 -tau_d=0.03  -algCudaVariant=1 -salMethod=0 -spMethod=1
// NLTV Sal optimzed version no energy
// NLTVSaliencyCuda.exe -lambda=0.1 -tau_p=0.3 -tau_d=0.03  -algCudaVariant=2 -salMethod=0 -spMethod=1 -tol=0
int main(int argc, char **argv)
{

	cv::CommandLineParser parser(argc, argv, keys);
	int boolAux;
	parser.about("NLTVSaliency CUDA v1.0.8");
	if (parser.has("help"))
	{
		parser.printMessage();
		return 0;
	}
	else {
		imagePath = parser.get<String>("imgName");
		superPixels = parser.get<int>("superPixels");
		compactness = parser.get<int>("compactness");
		
	

		locationPrior = parser.get<int>("locationPrior");
		locationPriorFlag = locationPrior == 1;


		boolAux = parser.get<int>("display");
		displayFlag = boolAux == 1;
		boolAux = parser.get<int>("write");
		writeFlag = boolAux == 1;
		execMethod = parser.get<int>("execMethod");


		
		red = parser.get<int>("red");
		green = parser.get<int>("green");
		blue = parser.get<int>("blue");
		imagePath = parser.get<String>("imgName");
		
		
		
		pathImages = parser.get<String>("pathImages");
		imgExtension = parser.get<String>("imgExtension");
		pathOut = parser.get<String>("pathOut");
		videoPath = parser.get<String>("videoPath");
		

	
		spMethod = parser.get<int>("spMethod");
		salMethod = parser.get<int>("salMethod");
		knn = parser.get<int>("knn");
		r = parser.get<float>("r");
		sigma2 = parser.get<float>("sigma2");
		/* Iterative Parameters */
		lambda = parser.get<float>("lambda");
		tau_p = parser.get<float>("tau_p");
		tau_d = parser.get<float>("tau_d");
		tol = parser.get<float>("tol");
		maxIter = parser.get<int>("maxIter");
		delta = parser.get<float>("delta");
		alpha = parser.get<float>("alpha");
		displayEnergyDebugFlag, displayImgDebugFlag;
		displayEnergyDebugFlag = parser.get<int>("displayEnergyDebug");
		displayImgDebugFlag = parser.get<int>("displayImgDebug");
		
		algCudaVariant = parser.get<int>("algCudaVariant");
		cudaDevice = parser.get<int>("cudaDevice");
		avg = parser.get<int>("avg");
		many = parser.get<int>("many");
		

		std::cout << "SpMethod " << spMethod  << " superPixels " << superPixels << "OpMethod " << salMethod<< std::endl;
		NLTVSaliencyAlg nltvSalAlg;
		NLTVSaliencyAlg::SaliencyParameters saliencyParameters;
		SlicParameters superpixelParameters(superPixels,red,green,blue,compactness,true);
		

		cudaDeviceProp deviceProp;
		cudaSetDevice(cudaDevice);
		cudaError_t error_id = cudaGetDeviceProperties(&deviceProp, cudaDevice);

		if (error_id == cudaSuccess)
		{
			printf(" Device %d: %s\n", cudaDevice, deviceProp.name);
		}
		saliencyParameters.r = r;
		saliencyParameters.sigma2 = sigma2;
		saliencyParameters.lambda = lambda;
		saliencyParameters.tau_p = tau_p;
		saliencyParameters.tau_d = tau_d;
		
		saliencyParameters.salMethod = static_cast<NLTVSaliencyAlg::SalMethod>(salMethod);
		saliencyParameters.spMethod = static_cast<SuperpixelAlgType>(spMethod);
		saliencyParameters.maxIter = maxIter;
		saliencyParameters.deltaSalTerm = delta;
		saliencyParameters.alphaSalTerm = alpha;
		saliencyParameters.energyDisplayOn = false;
		saliencyParameters.k = knn;
		saliencyParameters.algCudaVariant = static_cast<NLTVSaliencyAlg::AlgCudaVariant>(algCudaVariant);  
		
		saliencyParameters.tol = tol;
		saliencyParameters.imgDisplayOn = displayFlag;
		saliencyParameters.imgWrite = writeFlag;
		saliencyParameters.locationPrior = locationPriorFlag;


		/*superpixelParameters.compactness = compactness;
		superpixelParameters.sp = superPixels;
		superpixelParameters.red = red;
		superpixelParameters.red = red;
		superpixelParameters.green = green;
		superpixelParameters.blue = blue;*/
		
		print_parameters(saliencyParameters);
	
		float totalInMs, totalSlicTimeinMs = 0.0f, totalSaliencyTimeinMs = 0.0f;
		float slicTimeinMs, saliencyTimeinMs;
		switch (execMethod) {
		case 0:
#if BOOST_SUPPORT
			testReadFolder();
#endif

			break;
		case 1:
			if (avg == 0) {
				saliencyTimeinMs = nltvSalAlg.Execute(imagePath, superpixelParameters, saliencyParameters,&slicTimeinMs);
				cout << "\rSuperpixel in:[" << slicTimeinMs << "]ms" << endl;
				cout << "\rSaliencyAlg in:[" << saliencyTimeinMs  << "]ms" << endl;
				cout << "\rSaliency in:[" << (saliencyTimeinMs + slicTimeinMs) << "]ms" << endl;
				//getchar();
			}
			else {
				totalInMs = 0;
				saliencyParameters.imgDisplayOn = false;
				for (int i = 0; i< many; i++) {
					saliencyTimeinMs = nltvSalAlg.Execute(imagePath, superpixelParameters, saliencyParameters, &slicTimeinMs);
					totalSlicTimeinMs += slicTimeinMs;
					totalSaliencyTimeinMs += saliencyTimeinMs;
				}
				totalInMs = totalSlicTimeinMs + totalSaliencyTimeinMs;
				Mat imgColour;

				imgColour = cv::imread(imagePath, CV_LOAD_IMAGE_COLOR);
				//std::cout << "Total in Ms " << totalInMs << " each " << totalInMs / (float)many << " many " << many << std::endl;
				std::cout << "Image [" << imgColour.cols << "," << imgColour.rows << "]" << std::endl;
				std::cout << "Total in Ms " << totalInMs << " each " << totalInMs / (float)many << " many " << many << std::endl;
				std::cout << "SLIC in Ms " << totalSlicTimeinMs << " each " << totalSlicTimeinMs / (float)many << " many " << many << std::endl;
				std::cout << "Saliency in Ms " << totalSaliencyTimeinMs << " each " << totalSaliencyTimeinMs / (float)many << " many " << many << std::endl;
				//getchar();
			}
			break;

		}
		
	}
	
	
		
	return 0;
}