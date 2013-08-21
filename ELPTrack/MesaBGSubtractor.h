#ifndef _MESA_BG_SUBSTRACTOR_
#define _MESA_BG_SUBSTRACTOR_

#include <opencv2/core/core.hpp>


class MesaBGSubtractor
{
public:
	bool useAdaptive;
	cv::Mat convertedRange;
	cv::Mat background;
    cv::Mat foreground;
	cv::Mat difference;
	cv::Mat threshImage;
	bool firstFrame;

	float thresh;
	float adaptionRate;
	float nonAdaptionRate;

	MesaBGSubtractor(float adaptionRage, float thresh);
	~MesaBGSubtractor(void);

	void setAdaption(float f);
	void process(cv::Mat src, bool removeFromSrc);
	
};

#endif //_MESA_BG_SUBSTRACTOR_
