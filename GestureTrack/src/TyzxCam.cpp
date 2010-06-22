#include <iostream>
#include <sstream>
#include "TyzxCam.h"



TyzxCam::TyzxCam(const char *camIP, int port) {
	this->camIP = camIP;
	this->port = port;
	receiver = new TyzxTCPreceiver(camIP, port);

}

bool TyzxCam::start() {
	if(! receiver) {
		std::cerr << "TyzxCam::start()  -- TyzxTCPreceiver could not be created for " << camIP << " on port " << port << "\n";
		return false;
	} else {
		std::cout << "TyzxCam::start()  -- Initilizing Settings for " << camIP << " on port " << port << "\n";
	}

	receiver->initializeSettings();
	
	if(! receiver->probeConfiguration()) {
		std::cerr << "TyzxCam::start()  -- probeConfiguration failed for " << camIP << " on port " << port << "\n";
		return false;
	}
	int size = receiver->zHeight() * receiver->zWidth();
	points = new float[size*2*3];

	if (! receiver->startCapture()) {
		std::cerr << "TyzxCam::start()  -- startCapture failed for " << camIP << " on port " << port << "\n";
		return false;
	}

	return true;


}

bool TyzxCam::grab() {
	if(! receiver->grab()) {
		std::cerr << "TyzxCam::grab()  -- grab failed for " << camIP << " on port " << port << "\n";
		return false;
	}

	if(! receiver->isZEnabled()) {
		std::cerr << "TyzxCam::grab()  -- grab failed for " << camIP << " on port " << port << " range imaging was not enabled"<< "\n" ;
		return false;
	}

	zWidth = receiver->zWidth();
	zHeight = receiver->zHeight(); 
	zImage = receiver->getZImage();
	cx = receiver->getCx();
	cy = receiver->getCy();

	//std::cout << "cxyz " << cx << "  " << cy << "  "<< receiver->getCz() << " \n";

	receiver->getZimageCenter(imageCenterU, imageCenterV);

	return true;


}

//format
// 

//loat *TyzxCam::calcPoints() {
/*
	TyzxCam *zeroCam = cams[0];

	// assume both cams have same height and width.  Is this a safe assumption? TODO
	float zWidth = zeroCam->zWidth; 
	float zHeight = zeroCam->zHeight;


	size_t singleCamParamSize = sizeof(float) * 4;
	float* d_camParams;
	cutilSafeCall( cudaMalloc((void**)&d_zimg, fsize) ); // need to look up safecall TODO
	
	cutilSafeCall( cudaMalloc((void**)&d_zimg, fsize) ); // need to look up safecall TODO



	size_t singleZImageSize = zWidth * zHeight * sizeof(unsigned short);
	unsigned short* d_zimg;
	cutilSafeCall( cudaMalloc((void**)&d_zimg, fsize) ); // need to look up safecall TODO
	for(zeroCam

		
		
		size_t singlePointCloudSizeSize = zWidth * zHeight * sizeof(float);
	cutilSafeCall( cudaMemcpy(d_zimg, zImage, size, cudaMemcpyHostToDevice) );
	

	float* d_result;


	cutilSafeCall( cudaMalloc((void**)&d_result, ussize) );


	gpu_calcPointCloud(zWidth, zHeight, imageCenterU, imageCenterV, cx, cy, zimgs, result);
	/*
	int imageSize = zWidth * zHeight;
	unsigned short z;
	int u  = 0;
	int v =  0;


	float x;
	float y;

	double scaleU = 0;
	double scaleV = 0;


	int pointI = 0;

	for(int i = 0; i < imageSize; i++) {
		z = zImage[i];
		if ((z > 0) && (z < 65535)) { // if valid
			 x = (float) (((cx * u) - scaleU) * z);	
			 y = (float) (((cy * v) - scaleV) * z);

			 points[i++] = x;
			 points[i++] = y;
			 points[i++] = z;
		} else {
			 points[i++] = 0;
			 points[i++] = 0;
			 points[i++] = 0;
		}
		u++;
		if(u > zWidth) {
			u = 0;
			v++;
			scaleV = cy * v;
			scaleU = 0;
		} else {
			scaleV = cx * u;	
		}

	}

//	std::cout << "bounds "  << minBound << "  " << maxBound << "\n";
	return points;
	*/
//}

TyzxCam::~TyzxCam() {
	receiver->closeConnection();
	delete receiver;
}