
#include "../include/ISuperpixel.h"
#include "../include/SLICGpu.h"
#include "../include/SLICCpu.h"




/*static*/ SuperpixelAlgPtr CreateSuperpixelAlg(SuperpixelAlgType superpixelType)
{
	switch (superpixelType)
	{
	case SuperpixelAlgType::SLICGpu:
		return std::make_shared<SLICGpu>();
	case SuperpixelAlgType::SLICCpu:
		return std::make_shared<SLICCpu>();
	}

	throw new std::exception("Undefiend superpixel algorithm");
}