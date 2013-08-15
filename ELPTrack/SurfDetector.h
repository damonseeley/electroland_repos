#ifndef __SURF_DETECTOR__
#define __SURF_DETECTOR__
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/nonfree/features2d.hpp>

#define MINHESSIAN 500


//adapted from http://robocv.blogspot.com/2012/02/real-time-object-detection-in-opencv.html
//see also http://docs.opencv.org/doc/tutorials/features2d/feature_detection/feature_detection.html


class SurfDetector {
public:
	 cv::SurfFeatureDetector *detector;
	 cv::SurfDescriptorExtractor *extractor;
	cv::Mat object;
	cv::Mat des_object;
	 std::vector<cv::KeyPoint> kp_object;
	 cv::FlannBasedMatcher matcher;
	 std::vector<cv::Point2f> obj_corners;
	SurfDetector();
	void detect(cv::Mat src);
	~SurfDetector();
};
#endif
