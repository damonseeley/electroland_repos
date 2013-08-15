#include "FlyCap.h"

 #include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

 
#include <iostream>
 

FlyCap::FlyCap() {
	// Connect the camera
    error = camera.Connect( 0 );
    if ( error != PGRERROR_OK )
    {
        std::cout << "Failed to connect to camera" << std::endl;     
 //       return false;
    }
    
    // Get the camera info and print it out
    error = camera.GetCameraInfo( &camInfo );
    if ( error != PGRERROR_OK )
    {
        std::cout << "Failed to get camera info from camera" << std::endl;     
  //      return false;
    }
    std::cout << camInfo.vendorName << " "
    		  << camInfo.modelName << " " 
    		  << camInfo.serialNumber << std::endl;
	
	error = camera.StartCapture();
    if ( error == PGRERROR_ISOCH_BANDWIDTH_EXCEEDED )
    {
        std::cout << "Bandwidth exceeded" << std::endl;     
   //     return false;
    }
    else if ( error != PGRERROR_OK )
    {
        std::cout << "Failed to start image capture" << std::endl;     
   //     return false;
    } 
}

cv::Mat FlyCap::getImage() {
 
		Image rawImage;
		Error error = camera.RetrieveBuffer( &rawImage );
		if ( error != PGRERROR_OK )
		{
			std::cout << "capture error" << std::endl;
		}
		
		// convert to rgb
        rawImage.Convert( FlyCapture2::PIXEL_FORMAT_BGR, &rgbImage );
       
		// convert to OpenCV Mat
		unsigned int rowBytes = (double)rgbImage.GetReceivedDataSize()/(double)rgbImage.GetRows();       
		curImage = cv::Mat(rgbImage.GetRows(), rgbImage.GetCols(), CV_8UC3, rgbImage.GetData(),rowBytes);
//		cv::imshow("win", curImage);
//		cv::waitKey(30);
		return curImage;
		/*
		cv::Mat hsvImage;
		cv::Mat thresh;
		cv::cvtColor(image, hsvImage, CV_RGB2HSV);
		cv::inRange(hsvImage, cv::Scalar(0, 75,75), cv::Scalar(20,255,255), thresh);
		std::cout << "hsv " << (int) hsvImage.at<cv::Vec3b>(200,200)[0] << "," <<(int) hsvImage.at<cv::Vec3b>(200,200)[1] << ","<< (int)hsvImage.at<cv::Vec3b>(200,200)[2] << std::endl;
		std::cout << "rgb " << (int) image.at<cv::Vec3b>(200,200)[0] << "," << (int)image.at<cv::Vec3b>(200,200)[1] << ","<< (int)image.at<cv::Vec3b>(200,200)[2] << std::endl;
		cv::imshow("image", thresh);
		key = cv::waitKey(30);        
		*/
}

FlyCap::~FlyCap() {

	error = camera.StopCapture();
    if ( error != PGRERROR_OK )
    {
        // This may fail when the camera was removed, so don't show 
        // an error message
    }  
	
	camera.Disconnect();
	
}