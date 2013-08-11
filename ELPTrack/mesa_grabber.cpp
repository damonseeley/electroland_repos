/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 * $Id$
 *
 */

#include "mesa_grabber.h"


using namespace std;

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
MesaGrabber::MesaGrabber ()
{

	init();
	int result = SR_OpenDlg(&srCam, 2,0); // 2 -> opens a selection dialog
	if(result < 0) {
		 PCL_THROW_EXCEPTION (pcl::IOException, "[MesaGrabber::MesaGrabber] Error opening device from dialog");
	}
 
  
}

MesaGrabber::MesaGrabber (const char* filename_or_ipAddr, bool isIpAddress) //TODO add constructors for fixed IP and file
{
	init();
	int result;
	if(isIpAddress) {
		result =  SR_OpenETH(&srCam, filename_or_ipAddr);
	} else {
		result = SR_OpenFile(&srCam, filename_or_ipAddr);
	}

	if (result < 0) {
	  std::stringstream sstream;
	 sstream << "[MesaGrabber::MesaGrabber] Error opening device from " << filename_or_ipAddr;
	 PCL_THROW_EXCEPTION (pcl::IOException, sstream.str ());
	}

}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
MesaGrabber::~MesaGrabber () throw ()
{
  try
  {
    stop ();
	SR_Close(srCam);
  }
  catch (...)
  {
    // Destructor never throws
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
bool
MesaGrabber::isRunning () const
{
  return (running_);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
float
MesaGrabber::getFramesPerSecond () const
{ 
	return float(frameCnt) / float( pcl::getTime () - startTime);
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
MesaGrabber::init ()
{
//  setupDevice ();
  	frameCnt = 0;
  running_ = false;
  capture_thread_ = boost::thread (&MesaGrabber::captureThreadFunction, this);
}



//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
MesaGrabber::start ()
{
  frameCnt = 0;
  startTime = pcl::getTime();
  running_ = true;
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void
MesaGrabber::stop ()
{
  running_ = false;
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
pcl::PointCloud<pcl::PointXYZ>::Ptr
MesaGrabber::getXYZPointCloud ()
{

  
     // PCL point cloud
        pcl::PointCloud<pcl::PointXYZ>::Ptr
                cloud(new pcl::PointCloud<pcl::PointXYZ>(SR_GetRows(srCam), SR_GetCols(srCam)));
        
        float* ptrXYZ = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()
        int s = sizeof(pcl::PointXYZ); // size of a single PointXYZ structure for SR_CoordTrfFlt()
 
                SR_CoordTrfFlt(srCam, ptrXYZ, &ptrXYZ[1], &ptrXYZ[2], s, s, s); // convert to xyz
        

   return cloud;
}
pcl::PointCloud<pcl::PointXYZI>::Ptr
MesaGrabber::getXYZIPointCloud ()
{

  
     // PCL point cloud
        pcl::PointCloud<pcl::PointXYZI>::Ptr
                cloud(new pcl::PointCloud<pcl::PointXYZI>(SR_GetRows(srCam), SR_GetCols(srCam)));
        
        float* ptrXYZI = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()
        int s = sizeof(pcl::PointXYZI); // size of a single PointXYZI structure for SR_CoordTrfFlt()
		

              SR_CoordTrfFlt(srCam, ptrXYZI, &ptrXYZI[1], &ptrXYZI[2], s, s, 2*s); // convert to xyz
		     unsigned short* depthValues = (unsigned short*)SR_GetImage(srCam, 0);     // 0: return raw depth data
//			         float* ptrXYZI = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()

				for(int i = 0; i < cloud->points.size(); i++) {
					cloud->points[i].intensity = (float) depthValues[i];
				}
		
        

   return cloud;
}
pcl::PointCloud<pcl::PointXYZRGB>::Ptr
MesaGrabber::getXYZRGBPointCloud ()
{

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr
                cloud(new pcl::PointCloud<pcl::PointXYZRGB>(SR_GetRows(srCam), SR_GetCols(srCam)));
  //TODO
     // PCL point cloud
	// get xyz
	//make rgb
	// concat fields
	/*
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr
                cloud(new pcl::PointCloud<pcl::PointXYZRGB>(SR_GetRows(srCam), SR_GetCols(srCam)));
        
        float* ptrXYZ = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()
        int s = sizeof(pcl::PointXYZRGB); // size of a single PointXYZI structure for SR_CoordTrfFlt()
		

              SR_CoordTrfFlt(srCam, ptrXYZRGB, &ptrXYZRGB[1], &ptrXYZRGB[2], s, s, 4*s); // convert to xyz
		     unsigned short* depthValues = (unsigned short*)SR_GetImage(srCam, 0);     // 0: return raw depth data
//			         float* ptrXYZI = (float*)(&cloud->front()); // float pointer for SR_CoordTrfFlt()

				for(int i = 0; i < cloud->points.size(); i++) {
					ptrXYZRGB = points[i];
					cloud->points[i].r = (float) depthValues[i];
					cloud->points[i].g = (float) depthValues[i];
					cloud->points[i].b = (float) depthValues[i];
				}
		
        */

   return cloud;
}
void MesaGrabber::aquireImage() {
	               SR_Acquire(srCam); // get new depth image
 
}


//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
void 
MesaGrabber::captureThreadFunction ()
{
  while (true)
  {
    // Lock before checking running flag
    boost::unique_lock<boost::mutex> capture_lock (capture_mutex_);
    if(running_)
    {
    aquireImage();

      if (num_slots<sig_cb_mesa_point_cloud_xyzi> () > 0 )
        point_cloud_signal_xyzi->operator() (getXYZIPointCloud ()); 
      // Check for point clouds slots
      if (num_slots<sig_cb_mesa_point_cloud_xyz> () > 0 )
        point_cloud_signal_xyz->operator() (getXYZPointCloud ()); 
    } 
	frameCnt++;
    capture_lock.unlock ();
  }
}
