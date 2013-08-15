#include "SurfDetector.h"

#include <stdio.h>
#include <iostream>

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

SurfDetector::SurfDetector() {
	cv::Mat object = cv::imread( "/Users/electroland/Downloads/dtvLogo.jpg", CV_LOAD_IMAGE_GRAYSCALE );
	if( !object.data )
	{
		std::cout<< "Error image for object detection " << std::endl;
	}
	detector = new cv::SurfFeatureDetector (MINHESSIAN);
	extractor = new cv::SurfDescriptorExtractor();
	detector->detect( object, kp_object );


	extractor->compute( object, kp_object, des_object );
	obj_corners.resize(4);
	obj_corners[0] = cv::Point2f(0,0);
	obj_corners[1] = cv::Point2f( (float)object.cols, 0 );
	obj_corners[2] = cv::Point2f((float) object.cols,(float) object.rows );
	obj_corners[3] = cv::Point2f( 0,(float) object.rows );
}
void SurfDetector::detect(cv::Mat src) {
	cv::Mat des_image, img_matches;
	std::vector<cv::KeyPoint> kp_image;
	std::vector<std::vector<cv::DMatch > > matches;
	std::vector<cv::DMatch > good_matches;
	std::vector<cv::Point2f> obj;
	std::vector<cv::Point2f> scene;
	std::vector<cv::Point2f> scene_corners(4);
	cv::Mat H;
	cv:: Mat image;
	cvtColor(src, image, CV_RGB2GRAY);
	detector->detect( image, kp_image );
	extractor->compute( image, kp_image, des_image );
	matcher.knnMatch(des_object, des_image, matches, 2);
	//Get the corners from the object
	for(int i = 0; i < MIN(des_image.rows-1,(int) matches.size()); i++) //THIS LOOP IS SENSITIVE TO SEGFAULTS
	{
		if((matches[i][0].distance < 0.6*(matches[i][1].distance)) && ((int) matches[i].size()<=2 && (int) matches[i].size()>0))
		{
			good_matches.push_back(matches[i][0]);
		}
	}
	cv::drawMatches( object, kp_object, image, kp_image, good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );

	if (good_matches.size() >= 4)
	{
		for( int i = 0; i < good_matches.size(); i++ )
		{
			//Get the keypoints from the good matches
			obj.push_back( kp_object[ good_matches[i].queryIdx ].pt );
			scene.push_back( kp_image[ good_matches[i].trainIdx ].pt );
		}

		H = findHomography( obj, scene, CV_RANSAC );

		perspectiveTransform( obj_corners, scene_corners, H);

		//Draw lines between the corners (the mapped object in the scene image )
		line( img_matches, scene_corners[0] + cv::Point2f((float) object.cols, 0), scene_corners[1] + cv::Point2f( (float)object.cols, 0), cv::Scalar(0, 255, 0), 4 );
		line( img_matches, scene_corners[1] + cv::Point2f((float) object.cols, 0), scene_corners[2] + cv::Point2f((float) object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[2] + cv::Point2f( (float)object.cols, 0), scene_corners[3] + cv::Point2f( (float)object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
		line( img_matches, scene_corners[3] + cv::Point2f( (float)object.cols, 0), scene_corners[0] + cv::Point2f((float) object.cols, 0), cv::Scalar( 0, 255, 0), 4 );
	}

	//Show detected matches
	imshow( "Good Matches", img_matches );

	cv::waitKey(1);
}


SurfDetector::~SurfDetector() {
}
