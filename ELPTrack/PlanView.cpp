#include "PlanView.h"

//#include <opencv2/core/operations.hpp>
#include <opencv2/imgproc/imgproc.hpp>

PlanView::PlanView(float minX, float maxX, float minZ, float maxZ, int width, int height)
{
	setWorldDims(minX, maxX, minZ, maxZ);
	setBinDims(width, height);


	cv::SimpleBlobDetector::Params params;
	params.thresholdStep = 60;
	params.minThreshold = 60;
	params.maxThreshold = 255;
	params.minDistBetweenBlobs = 4.5;

	params.filterByColor = true;
	params.blobColor = 255;

	params.filterByArea = true;
	params.minArea = 4;
	params.maxArea = 100;

	params.filterByCircularity = false;

	params.filterByInertia = false;

	params.filterByConvexity=false;

	blobDetector = new cv::SimpleBlobDetector(params);
}


PlanView::~PlanView(void)
{
}

void PlanView::setWorldDims(float minX, float maxX, float minZ, float maxZ) {
	this->minX = minX;
	this->minZ = minZ;
	this->maxX = maxX;
	this->maxZ = maxZ;

	worldWidth = maxX-minX;
	worldHeight = maxZ-minZ;

	worldScaleX = binWidth/worldWidth;
	worldScaleZ = binHeight/worldHeight;

	worldScaleXInv = worldWidth/binWidth;
	worldScaleZInv = worldHeight/binHeight;

	minDetectZ = 1000;
	maxDetectZ =-1;


}
void PlanView::setBinDims(int width, int height){
	binWidth = width;
	binHeight = height;
	bins = cv::Mat(cv::Size(binWidth, binHeight), CV_32F);
	displayImage = cv::Mat(cv::Size(binWidth*3, binHeight*3),CV_8UC3) ;
}


void PlanView::worldDimsToBinDims(float x, float z, int &col, int &row) {
	col = (int)( (x-minX) * worldScaleX);
	row = (int)( (z-minZ) * worldScaleZ);
}

void PlanView::binDimsToWorldDims(float col, float row, float &x, float &z){
	x= (col * worldScaleXInv) + minX; 
	z= (row * worldScaleZInv) + minZ;

}

void PlanView::generatePlanView(pcl::PointCloud<pcl::PointXYZ>::Ptr pointCloud) {
	bins.setTo(cv::Scalar(0));

	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = pointCloud->points.begin(); it != pointCloud->points.end(); ++it) {
		int col;
		int row;
		worldDimsToBinDims(it->x, it->z, col, row);
		col = (col < 0) ? 0 : col;
		col = (col > binWidth - 1) ? binWidth -1: col;
		row = (row< 0) ? 0 : row;
		row = (row > binHeight-1) ? binHeight-1 : row;
		bins.at<float>(row, col)++;
	}

	// detect!
	bins.convertTo(blobsImage, CV_8UC1);

	// extract the x y coordinates of the keypoints: 


	//cv::dilate(thesh, thesh, 0, cv::Point(-1,-1), 3);
	//		double minVal;
	//	double maxVal;
	//cv::Point minLoc;
	//		cv::Point maxLoc;
	//	cv::minMaxLoc(bins, &minVal, &maxVal, &minLoc, &maxLoc);
	//std::cout << maxVal << " at " << maxLoc << std::endl;

	cv::threshold(bins, bins,2, 255, cv::THRESH_BINARY);
	bins.convertTo(thesh, CV_8UC1);

	cv::dilate(thesh, dilate1, cv::Mat(), cv::Point(-1,-1), 1);
	cv::erode(dilate1, dilate1, cv::Mat(), cv::Point(-1,-1), 1);

	cv::dilate(thesh, dilate2, cv::Mat(), cv::Point(-1,-1), 2);
	cv::erode(dilate2, dilate2, cv::Mat(), cv::Point(-1,-1), 2);

	//	cv::addWeighted(dilate1, .5, dilate2, .5, 0, dilate1);
	cv::addWeighted(dilate2, .5, thesh, .5, 0, thesh);

	blobDetector->detect(thesh, keypoints);

	cv::cvtColor(bins, blobsImage,CV_GRAY2RGB); 

//	cv::drawKeypoints(thesh, keypoints, blobsImage, cv::Scalar::all(-1));

	blobs.clear();

	for (int i=0; i<keypoints.size(); i++){
		float x;
		float z;
		binDimsToWorldDims(keypoints[i].pt.x, keypoints[i].pt.y, x,z);
//		float x= (keypoints[i].pt.x * worldScaleXInv) + minX; 
//		float z= (keypoints[i].pt.y * worldScaleZInv) + minZ;

		blobs.push_back(Blob(x,z));


	}
	/*for (int i=0; i<keypoints.size(); i++){
	float x=keypoints[i].pt.x; 
	float y=keypoints[i].pt.y;
	if(maxDetectZ < y) {
	std::cout << " maxZ " << maxDetectZ << std::endl;
	maxDetectZ = y;
	}
	if(minDetectZ > y) {
	std::cout << " minZ " << minDetectZ << std::endl;
	minDetectZ= y;
	}
	//	minDetectZ = minDetectZ < y? minDetectZ : y;
	//	maxDetectZ = minDetectZ < y? minDetectZ : y;
	//	std::cout << i << "\\" << keypoints.size() << "  (" << x << "," << y << ")"<<std::endl;
	}
	*/

		blobsImage.convertTo(blobsImage, CV_8UC3);
		cv::resize(blobsImage, displayImage, cv::Size(binWidth*3, binHeight*3));

}
