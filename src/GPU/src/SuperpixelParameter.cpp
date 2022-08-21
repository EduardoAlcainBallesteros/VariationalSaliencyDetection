#include "../include/SuperpixelParameters.h"


// 1. Default constructor
SuperpixelParameters::SuperpixelParameters()
{
}



SuperpixelParameters::SuperpixelParameters(int sp, int red, int green, int blue) {
	this->sp = sp;
	this->red = red;
	this->blue = blue;
	this->green = green;
}



// 2. Copy Constructor
SuperpixelParameters::SuperpixelParameters(const SuperpixelParameters & superpixelParameters) {
	this->sp = superpixelParameters.sp;
	this->red = superpixelParameters.red;
	this->blue = superpixelParameters.blue;
	this->green = superpixelParameters.green;
	
	
}
// 3. Destructor
// Virtual is important because we do not call the Specific Constructor 
// http://www.stroustrup.com/bs_faq2.html#virtual-dtor
//
SuperpixelParameters::~SuperpixelParameters() {
#ifdef NDEBUG
	std::cout << "SuperpixelParameters Call destructor"  << std::endl;
#endif
}

// 4. Assignment operator (operator =)
// http://en.cppreference.com/w/cpp/language/operators
SuperpixelParameters& SuperpixelParameters::operator =(const SuperpixelParameters& other) {
	if (this != &other) {
		this->sp = other.sp;
		this->red = other.red;
		this->blue = other.blue;
		this->green = other.green;

	}
	return *this;
}

int SuperpixelParameters::GetSp() const {
	return this->sp;
}
int SuperpixelParameters::GetRed() const {
	return this->red;
}
int SuperpixelParameters::GetGreen() const {
	return this->green;
}
int SuperpixelParameters::GetBlue() const {
	return this->blue;
}


