#include "PulseDetector.h"
	//list<TimeStampedMat> images;
	//long pluseWaveLength;

PulseDetector::PulseDetector(DWORD pulseWaveLength) {
	this->halfPulseWaveLength = pulseWaveLength * .5f;
}

cv::Mat PulseDetector::addImage(DWORD time, cv::Mat mat) {
	
	TimeStampedMat tsm;
	while ( (! images.empty()) && ((time - images.back().time) > halfPulseWaveLength)) {
		tsm = images.back();
		images.pop_back();
	}
	cv::Mat diff;
	cv::subtract(mat, tsm.mat, diff);
	tsm.mat = mat;
	tsm.time = time;
	images.push_front(tsm);
	return diff;
}


PulseDetector::~PulseDetector() {
	images.clear();
}
