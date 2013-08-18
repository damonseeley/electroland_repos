#include "PulseDetector.h"
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>

	//list<TimeStampedMat> images;
	//long pluseWaveLength;

PulseDetector::PulseDetector(DWORD colorWaveLength) {
	this->colorWaveLength = colorWaveLength;
	this->singleColorTime = (long) (colorWaveLength * .5f);
}

void PulseDetector::addImage(DWORD time, cv::Mat camImg) {
	cv::Mat hsv;
	cv::cvtColor(camImg, hsv, CV_RGB2HSV);

	TimeStampedMat curImg;
	curImg.time = time;

		cv::Scalar minRed(112, 200, 20);
		cv::Scalar maxRed(118, 255, 255);

		cv::Scalar minBlue(1, 200, 20);
		cv::Scalar maxBlue(5, 255, 255);


		cv::inRange(hsv, minRed, maxRed, curImg.red);
		cv::inRange(hsv, minBlue, maxBlue, curImg.blue);

	

		images.push_front(curImg);

	// get first frame that should be alternate color
	
	bool foundAlt = false;
	bool foundSame = false;
	TimeStampedMat altColor;
	TimeStampedMat sameColor;

	DWORD altTime = time- (1.0 * singleColorTime);
	DWORD sameTime = time-(1.0 * colorWaveLength);

	while(images.back().time < sameTime) {
		sameColor = images.back();
		images.pop_back();
		foundSame = true;
	}

	if(foundSame) {
		//pust image back on just incase its need again next frame
		images.push_back(sameColor);
		// start backwards looking for alternate color
		for(std::list<TimeStampedMat>::reverse_iterator it = images.rbegin(); it != images.rend(); it++) {
			if(it->time < altTime) {
				altColor = *it;
				foundAlt = true;
			} else {
				break;
			}
		}
	}
	foundPhone = false;
	if(foundSame && foundAlt) {
		foundPhone = true;

		
		cv::min(curImg.red, altColor.blue, scratch1);
		cv::min(curImg.blue, altColor.red, scratch2);
		cv::max(scratch1, scratch2, inPhase);

		cv::min(curImg.red, sameColor.red, scratch1);
		cv::min(curImg.blue, sameColor.blue, scratch2);
		cv::max(scratch1, scratch2, outOfPhase);


		cv::max(inPhase, outOfPhase, combo);
		//
	}

}



PulseDetector::~PulseDetector() {
	images.clear();
}
