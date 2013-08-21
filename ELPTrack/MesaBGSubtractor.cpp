#include "MesaBGSubtractor.h"

#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/video.hpp>
#include <opencv2/highgui/highgui.hpp>


MesaBGSubtractor::MesaBGSubtractor(void)
{
	firstFrame=true;
	useAdaptive = false;
	thresh = .075;
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
	cv::addWeighted(convertedRange, .001, background , .999, 0, background);
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