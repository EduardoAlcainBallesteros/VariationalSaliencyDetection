/*
* Simple wrapper class to measure time and return the result in different units see http://en.cppreference.com/w/cpp/chrono/duration
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
#ifndef __StopWatch_H__
#define __StopWatch_H__
#pragma once
#include <chrono>
//http://en.cppreference.com/w/cpp/chrono/duration
class StopWatch
{
public:
	enum TimeUnit
	{
		NANOSECONDS, MICROSECONDS, MILLISECONDS, SECONDS, MINUTES, HOURS
	};
	StopWatch(void);
	~StopWatch(void);
	/** Start counting the time
	*/
	void Start();
	/** Stop counting the time
	*/
	void Stop();
	/** Get the time between from start and stop interval
	*  @param timeUnit: units to return how much time past
	* return units elapsed from start to stop
	*/
	double GetElapsedTime(TimeUnit timeUnit)const;


private:
	const double SECOND_TO_MILLIS = 1000;
	const double MINUTE_TO_SECONDS = 60;
	const double HOUR_TO_MINUTES = 60;
	// // wraps QueryPerformanceCounter
	std::chrono::high_resolution_clock::time_point _startingTime;
	std::chrono::high_resolution_clock::time_point _endingTime;

};

#endif // __StopWatch_H__

