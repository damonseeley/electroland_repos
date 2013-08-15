#ifndef _PULSE_DETECTOR_
#define _PULSE_DETECTOR_
#include <Windows.h>
#include <list>
#include <opencv2/core/core.hpp>

class PulseDetector {
public:
	struct TimeStampedMat {
		DWORD time;
		cv::Mat mat;
	};

	std::list<TimeStampedMat> images;
	float halfPulseWaveLength;
	PulseDetector(DWORD pulseWaveLength);
	cv::Mat addImage(DWORD time, cv::Mat mat);
	~PulseDetector();
};


#endif