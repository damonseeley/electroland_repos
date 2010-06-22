#include "CloudConstructor.h"
#include <cuda_runtime.h>
#include <cutil_inline.h>
//#include <rendercheck_gl.h>


//#include<pointCloud.cu>

extern "C" void gpu_calcPointCloud(int camCnt, int imgWidth, int imgHeight, double* params, unsigned short *imgs, float* results);


CloudConstructor::CloudConstructor(TyzxCam *cams[], int camCnt) {
	this->cams = cams;
	this->camCnt = camCnt;
	camParams = new double[camCnt * 4];
	isInitNeeded = true;
}

void CloudConstructor::calcPoints() {

	if(isInitNeeded) {
		isInitNeeded = false;
		// assume all images have same dims 
		// TODO: is this safe?
		imgWidth = cams[0]->zWidth;
		imgHeight = cams[0]->zHeight;
		imgSize = imgWidth * imgHeight;

		//assume params dont' change after first grab
		// TODO: is this safe?
		double *ptParam = &camParams[0];
		for(int i = 0; i < camCnt; i++) {
			*ptParam++ = cams[i]->cx;
			*ptParam++ = cams[i]->cy;
			*ptParam++ = cams[i]->imageCenterU;
			*ptParam++ = cams[i]->imageCenterV;
		}
	}

	size_t paramSize = sizeof(double) * 4 * camCnt;
	double* d_fltParams;
	cutilSafeCall( cudaMalloc((void**)&d_fltParams, paramSize) ); // need to look up safecall TODO
	cutilSafeCall( cudaMemcpy(d_fltParams, camParams, paramSize, cudaMemcpyHostToDevice) );


	size_t singleZImageArrSize = imgSize * sizeof(unsigned short);
	float* d_zimg;
	cutilSafeCall( cudaMalloc((void**)&d_zimg, singleZImageArrSize * camCnt) ); // need to look up safecall TODO
	float *ptImage = &d_zimg[0];
	for(int i = 0; i < camCnt;i++) {
		cutilSafeCall( cudaMemcpy(ptImage, cams[i]->zImage, singleZImageArrSize, cudaMemcpyHostToDevice) );
		ptImage += imgSize;
	}


}


CloudConstructor::~CloudConstructor() {
}