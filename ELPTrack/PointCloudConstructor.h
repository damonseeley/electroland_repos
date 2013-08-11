#ifndef _PC_CONSTRUCTOR_
#define _PC_CONSTRUCTOR_
#define NOMINMAX
#include <Windows.h> // only on a Windows system
#undef NOMINMAX
//#include <pcl/console/parse.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <libMesaSR.h>


class PointCloudConstructor
{
public:
	EIGEN_MAKE_ALIGNED_OPERATOR_NEW

	typedef pcl::PointXYZ PointT;
	typedef pcl::PointCloud<PointT> PointCloudT;

	float* ptrXYZ;
	int xyzSize;

	SRCAM srCam;

	pcl::PointCloud<PointT>::Ptr cloud;

	pcl::PointCloud<PointT> filtered;
	pcl::PointCloud<PointT>::Ptr filteredPtr;

	pcl::PointCloud<PointT> transformed;
	pcl::PointCloud<PointT>::Ptr transformedPtr;

	Eigen::Affine3f floorTransform;


	PointCloudConstructor(SRCAM cam);
	~PointCloudConstructor(void);

	void setWorldDims(float minX, float maxX, float minY, float maxY, float minZ, float maxZ);
	void setTransform(float tx, float ty, float tz, float pitch, float yaw, float roll); 
	pcl::PointCloud<PointT>::Ptr  aquireFrame();
	pcl::PointCloud<PointT>::Ptr filterFrame();

	float minX,  minY,  minZ;
	float maxX,  maxY,   maxZ;
};
#endif //_PC_CONSTRUCTOR_

