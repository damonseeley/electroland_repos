#include "PointCloudConstructor.h"

#include <pcl/common/transforms.h>


PointCloudConstructor::PointCloudConstructor(SRCAM cam)
{
	srCam = cam;
	transformedPtr = pcl::PointCloud<PointT>::Ptr(&transformed);
	floorTransform= Eigen::Affine3f::Identity();
	filteredPtr =  pcl::PointCloud<PointT>::Ptr(&filtered);

	cloud = pcl::PointCloud<PointT>::Ptr(new pcl::PointCloud<PointT>(SR_GetRows(srCam), SR_GetCols(srCam)));      
	ptrXYZ = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()
	xyzSize = sizeof(PointT); // size of a single PointXYZ structure for SR_CoordTrfFlt()

//	box.setInputCloud(transformedPtr);

	minX = -2.2;
	maxX = 2;
	minY = 0;
	maxY = 2;
	minZ = 1.5;
	maxZ = 10;

	setWorldDims(minX, maxX, minY, maxY, minZ, maxZ);


}


PointCloudConstructor::~PointCloudConstructor(void)
{
}

void PointCloudConstructor::setTransform(float tx, float ty, float tz, float pitch, float yaw, float roll) {
	pcl::getTransformation (0,ty,0, pitch,yaw,roll, floorTransform);
}


pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudConstructor::aquireFrame() {
	SR_CoordTrfFlt(srCam, ptrXYZ, &ptrXYZ[1], &ptrXYZ[2],xyzSize,xyzSize,xyzSize); // convert to xyz
	pcl::transformPointCloud(*cloud,  transformed, floorTransform);
	return transformedPtr;
}
pcl::PointCloud<pcl::PointXYZ>::Ptr PointCloudConstructor::filterFrame() {
	
	filtered.clear();
	for(pcl::PointCloud<pcl::PointXYZ>::iterator it = transformedPtr->points.begin(); it != transformedPtr->points.end(); ++it) {
		if ( (it->x >= minX) && (it->x <= maxX) &&
			(it->y >= minY) && (it->y <= maxY) &&
			(it->z >= minZ) && (it->z <= maxZ) ) {
				PointT newPt(it->x, it->y, it->z);
				filtered.push_back(newPt);
		}
		//	std::cout << *it << "==?" << *it2 << std::endl;
	}
	
//	box.filter(filtered);

	return filteredPtr;
}


	void PointCloudConstructor::setWorldDims(float minX, float maxX, float minY, float maxY, float minZ, float maxZ) {
		this->minX = minX;
		this->maxX = maxX;
		this->minY = minY;
		this->maxY = maxY;
		this->minZ = minZ;
		this->maxZ = maxZ;

		/*
		Eigen::Vector4f minPoint;
		minPoint[0] = (this->minX < this->maxX) ? this->minX : this->maxX;
		minPoint[1] = (this->minY < this->maxY) ? this->minY : this->maxY;
		minPoint[2] = (this->minZ < this->maxZ) ? this->minZ : this->maxZ;
		box.setMin(minPoint);

		Eigen::Vector4f maxPoint;
		maxPoint[0] = (this->minX > this->maxX) ? this->minX : this->maxX;
		maxPoint[1] = (this->minY > this->maxY) ? this->minY : this->maxY;
		maxPoint[2] = (this->minZ > this->maxZ) ? this->minZ : this->maxZ;
		box.setMax(maxPoint);
		*/
	}
