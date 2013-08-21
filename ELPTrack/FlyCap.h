#ifndef __FLY_CAP__
#define __FLY_CAP__

#include <FlyCapture2.h>
#include <opencv2/core/core.hpp>

using namespace FlyCapture2;

class FlyCap {
public:
	Error error;
	Camera camera;
	CameraInfo camInfo;

	Image rgbImage;

	cv::Mat curImage;

	FlyCap();
	cv::Mat getImage();
	~FlyCap();
};
#endif