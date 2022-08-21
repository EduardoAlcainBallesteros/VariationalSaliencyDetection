#include "../include/SlicParameters.h"



SlicParameters::SlicParameters()
{
}

SlicParameters::SlicParameters(int sp, int red, int green, int blue, int compactness, int do_enforce_connectivity)
	: SuperpixelParameters(sp, red, green, blue) {
	this->compactness = compactness;
	this->do_enforce_connectivity = do_enforce_connectivity;
}

// 2. Copy Constructor
SlicParameters::SlicParameters(const SlicParameters & superpixelParameters) : SuperpixelParameters(superpixelParameters.GetSp(), superpixelParameters.GetRed(), superpixelParameters.GetGreen(), superpixelParameters.GetBlue()){
	this->compactness = superpixelParameters.compactness;
	this->do_enforce_connectivity = superpixelParameters.do_enforce_connectivity;


}

SlicParameters::~SlicParameters()
{
#ifdef NDEBUG
	std::cout << "SlicParameters Call destructor" << std::endl;
#endif
}


// 4. Assignment operator (operator =)
// http://en.cppreference.com/w/cpp/language/operators
SlicParameters& SlicParameters::operator =(const SlicParameters& other) {
	this->SuperpixelParameters::operator=(other);
	if (this != &other) {
		
		this->compactness = other.compactness;
		this->do_enforce_connectivity = other.do_enforce_connectivity;
	}
	return *this;
}

int SlicParameters::GetCompactness() const {
	return this->compactness;
}
int SlicParameters::GetDoEnforceConnectivity() const {
	return this->do_enforce_connectivity;
}
