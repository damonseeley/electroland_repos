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

#ifndef _MESA_GRABBER_
#define _MESA_GRABBER_
#define NOMINMAX
// windows import need to appear before libMesa or it won't compiel
#include <Windows.h> // only on a Windows system
#undef NOMINMAX
#include <libMesaSR.h>

#include <pcl/point_types.h>
#include <pcl/io/grabber.h>
#include <pcl/common/time.h>
#include <pcl/console/print.h>
#include <pcl/point_cloud.h>
#include <boost/circular_buffer.hpp>


class MesaGrabber: public pcl::Grabber
  {
    // Define callback signature typedefs

    typedef void (sig_cb_mesa_point_cloud_xyz) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZ> >&);
    typedef void (sig_cb_mesa_point_cloud_xyzi) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZI> >&);
    typedef void (sig_cb_mesa_point_cloud_xyzrgb) (const boost::shared_ptr<const pcl::PointCloud<pcl::PointXYZRGB> >&);
    
    public:
      /** \brief Constructor that sets up the grabber constants.
        * \param[in] device_position Number corresponding the device to grab
        */
	  MesaGrabber ();
      MesaGrabber (const char* filename_or_ipAddr, bool isIpAddress = true); //TODO add constructors for fixed IP and file

      /** \brief Destructor. It never throws. */
      virtual ~MesaGrabber () throw ();

      /** \brief Check if the grabber is running
        * \return true if grabber is running / streaming. False otherwise.
        */
      virtual bool 
      isRunning () const;
      
      /** \brief Returns the name of the concrete subclass, DinastGrabber.
        * \return DinastGrabber.
        */
      virtual std::string
      getName () const
      { return (std::string ("MesaGrabber")); }
      
      /** \brief Start the data acquisition process.
        */
      virtual void
      start ();

      /** \brief Stop the data acquisition process.
        */
      virtual void
      stop ();
      
      /** \brief Obtain the number of frames per second (FPS). */
      virtual float 
      getFramesPerSecond () const;

      
    protected:  
		SRCAM srCam; // SwissRanger

		double startTime;
		long frameCnt;
      
      void
      init ();  

	  void aquireImage();
      

      pcl::PointCloud<pcl::PointXYZ>::Ptr
      getXYZPointCloud ();
      
      pcl::PointCloud<pcl::PointXYZI>::Ptr
      getXYZIPointCloud ();
      pcl::PointCloud<pcl::PointXYZRGB>::Ptr
      getXYZRGBPointCloud ();
       /** \brief The function in charge of getting the data from the camera
        */     
      void 
      captureThreadFunction ();
      
      bool running_;
      boost::thread capture_thread_;
      
      mutable boost::mutex capture_mutex_;
      boost::signals2::signal<sig_cb_mesa_point_cloud_xyz>* point_cloud_signal_xyz;
      boost::signals2::signal<sig_cb_mesa_point_cloud_xyzi>* point_cloud_signal_xyzi;
      boost::signals2::signal<sig_cb_mesa_point_cloud_xyzrgb>* point_cloud_signal_xyzrgb;
  };

#endif // _MESA_GRABBER_
