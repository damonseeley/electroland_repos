#include "MesaBGSubtractor.h"

#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "ErrorLog.h"

MesaBGSubtractor::MesaBGSubtractor(float adaptionRate, float thresh)
{
	firstFrame=true;
	useAdaptive = false;
	this->thresh = thresh;
	this->setAdaption(adaptionRate);
}


MesaBGSubtractor::~MesaBGSubtractor(void)
{
}

void MesaBGSubtractor::process(cv::Mat src, bool removeFromSrc) {
	//	cv::cvtColor( src, src_gray, cv::COLOR_RGB2GRAY);
	if(useAdaptive) {
		src.convertTo(convertedRange, CV_8UC1, 1.0/256); // scale between 0-1

	} else {
		src.convertTo(convertedRange, CV_32F, 1.0/65536.0); // scale between 0-1
	}

	if(firstFrame) { 
		convertedRange.copyTo(background);
		firstFrame = false;
		if(useAdaptive) {
			background.convertTo(background, CV_8UC1);
			difference.convertTo(difference, CV_8UC1);
			threshImage.convertTo(threshImage, CV_8UC1);
		} else {
			threshImage.convertTo(threshImage, CV_32F);
			background.convertTo(background, CV_32F);
			difference.convertTo(difference, CV_32F);
		}
		
	}
	cv::addWeighted(convertedRange, adaptionRate, background , nonAdaptionRate, 0, background);
	cv::absdiff(convertedRange, background, difference);
	if(useAdaptive) {
		cv::adaptiveThreshold(difference, threshImage, 255, cv::ADAPTIVE_THRESH_MEAN_C, cv::THRESH_BINARY_INV, 21, -15);
	} else {
		cv::threshold(difference, threshImage, thresh, 1, cv::THRESH_BINARY_INV);
		threshImage.convertTo(threshImage, CV_8UC1, 255);
	}
	convertedRange.copyTo(foreground);
	foreground.setTo(cv::Scalar(0), threshImage);
	if(removeFromSrc)
		src.setTo(cv::Scalar(0), threshImage);

}

	void MesaBGSubtractor::setAdaption(float f) {
		if(f<0) {
			*ErrorLog::log << "Invalid background adaption rate " << f << ". it must be >= 0" << std::endl;
			f = 0;
		}
		if(f>1) {
			*ErrorLog::log << "Invalid background adaption rate " << f << ". it must be <= 1" << std::endl;
			f = 1;
		}

		adaptionRate = f;
		nonAdaptionRate = 1.0-f;
	}
