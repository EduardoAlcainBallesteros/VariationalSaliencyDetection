#include "../include/SLICGpu.h"

void  SLICGpu::SetSettings(const SuperpixelParameters&  superpixelParameters) {
	const SlicParameters & s = dynamic_cast<const SlicParameters&>(superpixelParameters);
	this->slicParameters = s;
}


void SLICGpu::mat_to_uchar4Image(const cv::Mat& inimg, gSLICr::UChar4Image* outimg)
{
	gSLICr::Vector4u* outimg_ptr = outimg->GetData(MEMORYDEVICE_CPU);

	for (int y = 0; y < outimg->noDims.y; y++)
		for (int x = 0; x < outimg->noDims.x; x++)
		{
			int idx = x + y * outimg->noDims.x;
			outimg_ptr[idx].b = inimg.at<cv::Vec3b>(y, x)[0];
			outimg_ptr[idx].g = inimg.at<cv::Vec3b>(y, x)[1];
			outimg_ptr[idx].r = inimg.at<cv::Vec3b>(y, x)[2];
		}
}


void SLICGpu::settings_SLIC(const int cols, const int rows, gSLICr::objects::settings & settings) {
	int spixel_size = 120;
	settings.img_size.x = cols;
	settings.img_size.y = rows;
	settings.no_segs = this->slicParameters.GetSp();
	settings.spixel_size = spixel_size;

	int sz = cols * rows;
	int step = sqrt((double)(sz) / (double)(this->slicParameters.GetSp())) + 0.5;
	float invwt = (1.0 / ((step / (float)this->slicParameters.GetCompactness())*(step / (float)this->slicParameters.GetCompactness()))) / 0.0045f;

	//my_settings.coh_weight = invwt;
	settings.coh_weight = this->slicParameters.GetCompactness();
	std::cout << "Compactness " << this->slicParameters.GetCompactness() << " Invwt " << invwt << " coh_weight " << settings.coh_weight << std::endl;
	settings.no_iters = 10;
	settings.color_space = gSLICr::CIELAB; //gSLICr::XYZ gSLICr::CIELAB for Lab, or gSLICr::RGB for RGB
	settings.seg_method = gSLICr::GIVEN_NUM; // gSLICr::GIVEN_SIZE; // or gSLICr::GIVEN_NUM for given number
	settings.do_enforce_connectivity = true; // wheter or not run the enforce connectivity step
}


void SLICGpu::SuperPixelSegmentation(const cv::Mat& image)

{
	
	StopWatchInterface *slicTimer;
	// gSLICr settings
	gSLICr::objects::settings slic_settings;
	settings_SLIC(image.cols, image.rows, slic_settings);
	int size = image.cols *image.rows;
	std::cout << "SLIC GPU " << this->slicParameters.GetSp() << " Compactness " << slicParameters.GetCompactness()<< std::endl;
	sdkCreateTimer(&slicTimer);

	sdkResetTimer(&slicTimer); sdkStartTimer(&slicTimer);
	// instantiate a core_engine
	gSLICr::engines::core_engine* gSLICr_engine = new gSLICr::engines::core_engine(slic_settings);

	// gSLICr takes gSLICr::UChar4Image as input and out put
	gSLICr::UChar4Image* in_img = new gSLICr::UChar4Image(slic_settings.img_size, true, true);
	gSLICr::UChar4Image* out_img = new gSLICr::UChar4Image(slic_settings.img_size, true, true);

	mat_to_uchar4Image(image, in_img);


	gSLICr_engine->Process_Frame(in_img);

	sdkStopTimer(&slicTimer);
	timeInMs = sdkGetTimerValue(&slicTimer);


	const gSLICr::IntImage* labelsSLIC = gSLICr_engine->GetMask();
	const int * dev_labels = labelsSLIC->GetData(MemoryDeviceType::MEMORYDEVICE_CUDA);
	const int * h_labels = labelsSLIC->GetData(MemoryDeviceType::MEMORYDEVICE_CPU);

	outputNumSuperpixels = 1 + maxElements<int>(size, dev_labels);
	std::cout << "NumSuperpixels " << outputNumSuperpixels << std::endl;
	clabels = (int *)malloc(sizeof(int)* size);
	checkCudaErrors(cudaMemcpy(clabels, dev_labels, size * sizeof(*clabels), cudaMemcpyDeviceToHost));

	delete in_img;
	delete out_img;

	delete gSLICr_engine;
	

	sdkDeleteTimer(&slicTimer);
	

}
float SLICGpu::GetTimeInMs() {
	return timeInMs;
}


SLICGpu::SLICGpu()
{
	clabels = NULL;
	outputNumSuperpixels = 0;

}


SLICGpu::~SLICGpu()
{
#ifdef NDEBUG	
	printf("~SLICGpu\n");
#endif
	if (NULL != clabels) {
		free(clabels);
		clabels = NULL;
	}
}
int SLICGpu::GetNumSuperpixels() {
	return outputNumSuperpixels;

}




int * SLICGpu::GetLabels() {
	return clabels;
}
std::string SLICGpu::Type()
{
	return std::string("SLIC");
}

