#ifndef _PULSE_DETECTOR_
#define _PULSE_DETECTOR_
#include <Windows.h>
#include <list>
#include <opencv2/core/core.hpp>

class PulseDetector {
public:

		cv::Mat displayImage;
			

//		cv::Mat diffRGB;
//		cv::Mat sameRGB;
		

		cv::Mat scratch1;
		cv::Mat scratch2;

		cv::Mat inPhase;
		cv::Mat outOfPhase;
		
		
		
		cv::Mat combo;
		bool foundPhone;


	struct TimeStampedMat {
		DWORD time;
		cv::Mat red;
		cv::Mat blue;
	};

	std::list<TimeStampedMat> images;
	DWORD colorWaveLength; // (time for full oscillation, frequency should never be more than half effective fps)
	DWORD singleColorTime; // (.5 * colorWaveLength)
	PulseDetector(DWORD colorWaveLength);
	void addImage(DWORD time, cv::Mat mat);
	~PulseDetector();
};


#endif