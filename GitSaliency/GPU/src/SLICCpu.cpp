#include "../include/SLICCpu.h"

void  SLICCpu::SetSettings(const SuperpixelParameters&  superpixelParameters) {
	const SlicParameters & s = dynamic_cast<const SlicParameters&>(superpixelParameters);
	this->slicParameters = s;
}

template<typename T>
void SLICCpu::rgbtolab(int* rin, int* gin, int* bin, int sz, T* lvec, T* avec, T* bvec)
{
	int i; int sR, sG, sB;
	double R, G, B;
	double X, Y, Z;
	double r, g, b;
	const double epsilon = 0.008856;	//actual CIE standard
	const double kappa = 903.3;		//actual CIE standard

	const double Xr = 0.950456;	//reference white
	const double Yr = 1.0;		//reference white
	const double Zr = 1.088754;	//reference white
	double xr, yr, zr;
	double fx, fy, fz;
	double lval, aval, bval;

	for (i = 0; i < sz; i++)
	{
		sR = rin[i]; sG = gin[i]; sB = bin[i];
		R = sR / 255.0;
		G = sG / 255.0;
		B = sB / 255.0;

		if (R <= 0.04045)	r = R / 12.92;
		else				r = pow((R + 0.055) / 1.055, 2.4);
		if (G <= 0.04045)	g = G / 12.92;
		else				g = pow((G + 0.055) / 1.055, 2.4);
		if (B <= 0.04045)	b = B / 12.92;
		else				b = pow((B + 0.055) / 1.055, 2.4);

		X = r*0.4124564 + g*0.3575761 + b*0.1804375;
		Y = r*0.2126729 + g*0.7151522 + b*0.0721750;
		Z = r*0.0193339 + g*0.1191920 + b*0.9503041;

		//------------------------
		// XYZ to LAB conversion
		//------------------------
		xr = X / Xr;
		yr = Y / Yr;
		zr = Z / Zr;

		if (xr > epsilon)	fx = pow(xr, 1.0 / 3.0);
		else				fx = (kappa*xr + 16.0) / 116.0;
		if (yr > epsilon)	fy = pow(yr, 1.0 / 3.0);
		else				fy = (kappa*yr + 16.0) / 116.0;
		if (zr > epsilon)	fz = pow(zr, 1.0 / 3.0);
		else				fz = (kappa*zr + 16.0) / 116.0;

		lval = 116.0*fy - 16.0;
		aval = 500.0*(fx - fy);
		bval = 200.0*(fy - fz);

		lvec[i] = lval; avec[i] = aval; bvec[i] = bval;
	}
}


void SLICCpu::getLABXYSeeds(int STEP, int width, int height, int* seedIndices, int* numseeds)
{
	const bool hexgrid = false;
	int n;
	int xstrips, ystrips;
	int xerr, yerr;
	double xerrperstrip, yerrperstrip;
	int xoff, yoff;
	int x, y;
	int xe, ye;
	int seedx, seedy;
	int i;

	xstrips = (0.5 + (double)(width) / (double)(STEP));
	ystrips = (0.5 + (double)(height) / (double)(STEP));

	xerr = width - STEP*xstrips; if (xerr < 0) { xstrips--; xerr = width - STEP*xstrips; }
	yerr = height - STEP*ystrips; if (yerr < 0) { ystrips--; yerr = height - STEP*ystrips; }

	xerrperstrip = (double)(xerr) / (double)(xstrips);
	yerrperstrip = (double)(yerr) / (double)(ystrips);

	xoff = STEP / 2;
	yoff = STEP / 2;

	n = 0;
	for (y = 0; y < ystrips; y++)
	{
		ye = y*yerrperstrip;
		for (x = 0; x < xstrips; x++)
		{
			xe = x*xerrperstrip;
			seedx = (x*STEP + xoff + xe);
			if (hexgrid) { seedx = x*STEP + (xoff << (y & 0x1)) + xe; if (seedx >= width)seedx = width - 1; }//for hex grid sampling
			seedy = (y*STEP + yoff + ye);
			i = seedy*width + seedx;
			seedIndices[n] = i;
			n++;
		}
	}
	*numseeds = n;
}
template<typename T>

void SLICCpu::PerformSuperpixelSLIC(T* lvec, T* avec, T* bvec, T* kseedsl, T* kseedsa, T* kseedsb, T* kseedsx, T* kseedsy, int width, int height, int numseeds, int* klabels, int STEP, double compactness)
{
	int x1, y1, x2, y2;
	double l, a, b;
	double dist;
	double distxy;
	int itr;
	int n;
	int x, y;
	int i;
	int ind;
	int r, c;
	int k;
	int sz = width*height;
	const int numk = numseeds;
	int offset = STEP;

	T* clustersize = (T*)malloc(sizeof(T)*numk);
	T* inv = (T*)malloc(sizeof(T)*numk);
	T* sigmal = (T*)malloc(sizeof(T)*numk);
	T* sigmaa = (T*)malloc(sizeof(T)*numk);
	T* sigmab = (T*)malloc(sizeof(T)*numk);
	T* sigmax = (T*)malloc(sizeof(T)*numk);
	T* sigmay = (T*)malloc(sizeof(T)*numk);
	T* distvec = (T*)malloc(sizeof(T)*sz);
	double invwt = 1.0 / ((STEP / compactness)*(STEP / compactness));
	// 10 Iterations mentioned in the paper
	for (itr = 0; itr < 10; itr++)
	{
		for (i = 0; i < sz; i++) { distvec[i] = DBL_MAX; }

		for (n = 0; n < numk; n++)
		{
			x1 = kseedsx[n] - offset; if (x1 < 0) x1 = 0;
			y1 = kseedsy[n] - offset; if (y1 < 0) y1 = 0;
			x2 = kseedsx[n] + offset; if (x2 > width)  x2 = width;
			y2 = kseedsy[n] + offset; if (y2 > height) y2 = height;

			for (y = y1; y < y2; y++)
			{
				for (x = x1; x < x2; x++)
				{
					i = y*width + x;

					l = lvec[i];
					a = avec[i];
					b = bvec[i];

					dist = (l - kseedsl[n])*(l - kseedsl[n]) +
						(a - kseedsa[n])*(a - kseedsa[n]) +
						(b - kseedsb[n])*(b - kseedsb[n]);

					distxy = (x - kseedsx[n])*(x - kseedsx[n]) + (y - kseedsy[n])*(y - kseedsy[n]);

					dist += distxy*invwt;

					if (dist < distvec[i])
					{
						distvec[i] = dist;
						klabels[i] = n;
					}
				}
			}
		}
		//-----------------------------------------------------------------
		// Recalculate the centroid and store in the seed values
		//-----------------------------------------------------------------
		/*for (k = 0; k < numk; k++)
		{
		sigmal[k] = 0;
		sigmaa[k] = 0;
		sigmab[k] = 0;
		sigmax[k] = 0;
		sigmay[k] = 0;
		clustersize[k] = 0;
		}*/
		// Optimization 
		memset(sigmal, 0, sizeof(T)*numk);
		memset(sigmaa, 0, sizeof(T)*numk);
		memset(sigmab, 0, sizeof(T)*numk);
		memset(sigmax, 0, sizeof(T)*numk);
		memset(sigmay, 0, sizeof(T)*numk);
		memset(clustersize, 0, sizeof(T)*numk);

		ind = 0;
		for (r = 0; r < height; r++)
		{
			for (c = 0; c < width; c++)
			{
				if (klabels[ind] >= 0)
				{
					sigmal[klabels[ind]] += lvec[ind];
					sigmaa[klabels[ind]] += avec[ind];
					sigmab[klabels[ind]] += bvec[ind];
					sigmax[klabels[ind]] += c;
					sigmay[klabels[ind]] += r;
					clustersize[klabels[ind]] += 1.0;
				}
				ind++;
			}
		}

		{for (k = 0; k < numk; k++)
		{
			if (clustersize[k] <= 0) clustersize[k] = 1;
			inv[k] = 1.0 / clustersize[k];//computing inverse now to multiply, than divide later
		}}

		{for (k = 0; k < numk; k++)
		{
			kseedsl[k] = sigmal[k] * inv[k];
			kseedsa[k] = sigmaa[k] * inv[k];
			kseedsb[k] = sigmab[k] * inv[k];
			kseedsx[k] = sigmax[k] * inv[k];
			kseedsy[k] = sigmay[k] * inv[k];
		}}
	}// for iter
	free(sigmal);
	free(sigmaa);
	free(sigmab);
	free(sigmax);
	free(sigmay);
	free(clustersize);
	free(inv);
	free(distvec);

}

void SLICCpu::EnforceSuperpixelConnectivity(int* labels, int width, int height, int numSuperpixels, int* nlabels, int* finalNumberOfLabels)
{
	int i, j, k;
	int n, c, count;
	int x, y;
	int ind;
	int oindex, adjlabel;
	int label;
	const int dx4[4] = { -1, 0, 1, 0 };
	const int dy4[4] = { 0, -1, 0, 1 };
	const int sz = width*height;
	const int SUPSZ = sz / numSuperpixels;
	int* xvec = (int*)malloc(sizeof(int)*SUPSZ * 10);
	int* yvec = (int*)malloc(sizeof(int)*SUPSZ * 10);

	for (i = 0; i < sz; i++) nlabels[i] = -1;
	oindex = 0;
	adjlabel = 0;//adjacent label
	label = 0;
	for (j = 0; j < height; j++)
	{
		for (k = 0; k < width; k++)
		{
			if (0 > nlabels[oindex])
			{
				nlabels[oindex] = label;
				//--------------------
				// Start a new segment
				//--------------------
				xvec[0] = k;
				yvec[0] = j;
				//-------------------------------------------------------
				// Quickly find an adjacent label for use later if needed
				//-------------------------------------------------------
				{for (n = 0; n < 4; n++)
				{
					int x = xvec[0] + dx4[n];
					int y = yvec[0] + dy4[n];
					if ((x >= 0 && x < width) && (y >= 0 && y < height))
					{
						int nindex = y*width + x;
						if (nlabels[nindex] >= 0) adjlabel = nlabels[nindex];
					}
				}}

				count = 1;
				for (c = 0; c < count; c++)
				{
					for (n = 0; n < 4; n++)
					{
						x = xvec[c] + dx4[n];
						y = yvec[c] + dy4[n];

						if ((x >= 0 && x < width) && (y >= 0 && y < height))
						{
							int nindex = y*width + x;

							if (0 > nlabels[nindex] && labels[oindex] == labels[nindex])
							{
								xvec[count] = x;
								yvec[count] = y;
								nlabels[nindex] = label;
								count++;
							}
						}

					}
				}
				//-------------------------------------------------------
				// If segment size is less then a limit, assign an
				// adjacent label found before, and decrement label count.
				//-------------------------------------------------------
				if (count <= SUPSZ >> 2)
				{
					for (c = 0; c < count; c++)
					{
						ind = yvec[c] * width + xvec[c];
						nlabels[ind] = adjlabel;
					}
					label--;
				}
				label++;
			}
			oindex++;
		}
	}
	*finalNumberOfLabels = label;

	free(xvec);
	free(yvec);
}



void SLICCpu::SuperPixelSegmentation(const cv::Mat& image)
{	
	StopWatch stopWatch;
		int sz;
		int i, ii;
		int x, y;
		int* rin; int* gin; int* bin;
		int* klabels;
		//int* clabels;
		float* lvec; float* avec; float* bvec;
		int step;
		int* seedIndices;
		int numseeds;
		float* kseedsx; float* kseedsy;
		float* kseedsl; float* kseedsa; float* kseedsb;
		int k;
		int width, height, channels;

		//int* outputNumSuperpixels;
		int* outlabels;
		int finalNumberOfLabels;
		const unsigned char* imgbytes;
		
		width = image.cols;
		height = image.rows;
		channels = image.channels();
		imgbytes = NULL;

		//imgbytes = (unsigned char*)mxGetData(prhs[0]);//mxGetData returns a void pointer, so cast it
		//width = dims[1]; height = dims[0];//Note: first dimension provided is height and second is width
		sz = width*height;
		//---------------------------
		
		stopWatch.Start();

		//---------------------------
		// Allocate memory
		//---------------------------
		rin = (int*)malloc(sizeof(int)      * sz);
		gin = (int*)malloc(sizeof(int)      * sz);
		bin = (int*)malloc(sizeof(int)      * sz);
		lvec = (float*)malloc(sizeof(float)      * sz);
		avec = (float*)malloc(sizeof(float)      * sz);
		bvec = (float*)malloc(sizeof(float)      * sz);
		klabels = (int*)malloc(sizeof(int)         * sz);//original k-means labels
		seedIndices = (int*)malloc(sizeof(int)     * sz);
		outputNumSuperpixels = 0;
		if (clabels != NULL) {
			free(clabels);
		}
		clabels = (int*)malloc(sizeof(int)*sz);

		//---------------------------
		// Perform color conversion
		//---------------------------
		//if(2 == numdims)
		if (channels == 1)//if it is a grayscale image, copy the values directly into the lab vectors
		{
			imgbytes = image.ptr<uchar>(0);

			for (i = 0; i < sz; i++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
			{

				lvec[i] = imgbytes[i];
				avec[i] = imgbytes[i];
				bvec[i] = imgbytes[i];
			}
		}
		else//else covert from rgb to lab
		{
			const cv::Vec3b *imgColour = image.ptr<cv::Vec3b>(0);
			for (i = 0; i < sz; i++)//reading data from column-major MATLAB matrics to row-major C matrices (i.e perform transpose)
			{
				uchar blue = imgColour[i].val[0];
				uchar green = imgColour[i].val[1];
				uchar red = imgColour[i].val[2];
				rin[i] = red;
				gin[i] = green;
				bin[i] = blue;

			}
			rgbtolab(rin, gin, bin, sz, lvec, avec, bvec);
		}
		//---------------------------
		// Find seeds
		//---------------------------
		step = sqrt((double)(sz) / (double)(slicParameters.GetSp())) + 0.5;
		getLABXYSeeds(step, width, height, seedIndices, &numseeds);

		kseedsx = (float*)malloc(sizeof(float)      * numseeds);
		kseedsy = (float*)malloc(sizeof(float)      * numseeds);
		kseedsl = (float*)malloc(sizeof(float)      * numseeds);
		kseedsa = (float*)malloc(sizeof(float)      * numseeds);
		kseedsb = (float*)malloc(sizeof(float)      * numseeds);
		for (k = 0; k < numseeds; k++)
		{
			kseedsx[k] = seedIndices[k] % width;
			kseedsy[k] = seedIndices[k] / width;
			kseedsl[k] = lvec[seedIndices[k]];
			kseedsa[k] = avec[seedIndices[k]];
			kseedsb[k] = bvec[seedIndices[k]];
		}
		//---------------------------
		// Compute superpixels
		//---------------------------
		PerformSuperpixelSLIC(lvec, avec, bvec, kseedsl, kseedsa, kseedsb, kseedsx, kseedsy, width, height, numseeds, klabels, step, slicParameters.GetCompactness());
		
		//---------------------------
		// Enforce connectivity
		//---------------------------
		EnforceSuperpixelConnectivity(klabels, width, height,slicParameters.GetSp(), clabels, &finalNumberOfLabels);
		



		outputNumSuperpixels = finalNumberOfLabels;
	

	

		//---------------------------
		// Deallocate memory
		//---------------------------
		free(rin);
		free(gin);
		free(bin);
		free(lvec);
		free(avec);
		free(bvec);
		free(klabels);

		free(seedIndices);
		free(kseedsx);
		free(kseedsy);
		free(kseedsl);
		free(kseedsa);
		free(kseedsb);

	
		stopWatch.Stop();

		timeInMs = stopWatch.GetElapsedTime(StopWatch::MILLISECONDS);


}
float SLICCpu::GetTimeInMs() {
	return timeInMs;
}


SLICCpu::SLICCpu()
{
	clabels = NULL;
	outputNumSuperpixels = 0;

}


SLICCpu::~SLICCpu()
{
#ifdef NDEBUG	
	printf("~SLICGpu\n");
#endif
	if (NULL != clabels) {
		free(clabels);
		clabels = NULL;
	}
}
int SLICCpu::GetNumSuperpixels() {
	return outputNumSuperpixels;

}




int * SLICCpu::GetLabels() {
	return clabels;
}
std::string SLICCpu::Type()
{
	return std::string("SLICCpu");
}


