#ifndef _PLAN_VIEW_
#define _PLAN_VIEW_

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include "Blob.h"

class PlanView
{
public:
	cv::Mat bins;
	cv::Mat thesh;
	cv::Mat dilate1;
	cv::Mat dilate2;
	cv::Mat blobsImage;
	cv::Mat displayImage;
	cv::SimpleBlobDetector *blobDetector;
	
	int blurRadius;
	float pointCntThresh;

	std::vector<Blob> blobs;


	bool flipX;
	bool flipZ;


	float minX;
	float maxX;
	float minZ;
	float maxZ;
	float worldWidth;
	float worldHeight;

	float worldScaleX; // world to bins
	float worldScaleZ;

	float worldScaleXInv; // bins to world
	float worldScaleZInv;

	int binWidth;
	int binHeight;

	float minDetectZ;
	float maxDetectZ;

	std::vector<cv::KeyPoint> keypoints;


	PlanView(float minX, float maxX, float minZ, float maxZ, int width, int height);
	~PlanView(void);

	void worldDimsToBinDims(float x, float z, int &col, int &row);
	void binDimsToWorldDims(float col, float row, float &x, float &z);

	void setWorldDims(float minX, float maxX, float minZ, float maxZ);
	void setBinDims(int width, int height);


	void generatePlanView(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud);

	void setFlipX(bool b);
	void setFlipZ(bool b);

};

#endif
