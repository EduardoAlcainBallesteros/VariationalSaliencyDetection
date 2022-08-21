#include "../include/StopWatch.h"


StopWatch::StopWatch() {

}

StopWatch::~StopWatch() {

}

void StopWatch::Start() {
	_startingTime = std::chrono::high_resolution_clock::now();
}

void StopWatch::Stop() {
	_endingTime = std::chrono::high_resolution_clock::now();
}
//http://en.cppreference.com/w/cpp/chrono/duration/duration_cast
double StopWatch::GetElapsedTime(TimeUnit timeUnit) const {
	double elapsedTime;

	//http://en.cppreference.com/w/cpp/numeric/ratio/ratio
	std::chrono::duration<double, std::milli> fp_ms;
	std::chrono::duration<double, std::nano> fp_nano;
	std::chrono::duration<double, std::micro> fp_mc;
	std::chrono::duration<double, std::deca> fp_deca;
	switch (timeUnit)
	{
	case NANOSECONDS:
		fp_nano = _endingTime - _startingTime;
		elapsedTime = fp_nano.count();
		break;
	case MICROSECONDS:
		fp_mc = _endingTime - _startingTime;
		elapsedTime = fp_mc.count();
		break;
	case MILLISECONDS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = fp_ms.count();
		break;
	case SECONDS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / SECOND_TO_MILLIS;
		break;
	case MINUTES:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / (SECOND_TO_MILLIS * MINUTE_TO_SECONDS);
		break;
	case HOURS:
		fp_ms = _endingTime - _startingTime;
		elapsedTime = (double)fp_ms.count() / (SECOND_TO_MILLIS * MINUTE_TO_SECONDS * HOUR_TO_MINUTES);
		break;
	default:
		elapsedTime = -1;
		break;
	}

	return elapsedTime;
}
