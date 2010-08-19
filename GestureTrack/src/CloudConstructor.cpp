#include "CloudConstructor.h"
#include <cuda_runtime.h>
#include <cutil_inline.h>

//#include<pointCloud.cu>


#include <stdio.h>
#include <iostream>

//#define USE_CPU

extern "C" void gpu_calcPointCloud(int camCnt, int imgWidth, int imgHeight, double* params, double* transforms, unsigned short *imgs, float* results);

CloudConstructor::CloudConstructor(TyzxCam *cams[], int camCnt) {
	this->cams = cams;
	this->camCnt = camCnt;
	//	camParams = (double *)malloc(4*camCnt*sizeof(double)); // this would be better witha typedef
	camParams = new double[camCnt * CAM_PARAM_CNT];
	isInitNeeded = true;
}



float* CloudConstructor::getGPUPoints() {
	if(d_resultPoints == NULL) {
		size_t resultSize = pointCnt * 3 * sizeof(float);
		cutilSafeCall( cudaMalloc((void**)&d_resultPoints, resultSize));
		cutilSafeCall(cudaMemcpy(d_resultPoints, points, resultSize, cudaMemcpyHostToDevice));
	}
		return d_resultPoints;
	
}
void CloudConstructor::freeGPUPoints() {
	cudaFree(d_resultPoints);
	d_resultPoints = NULL;
}

// points only need to be cudaMalloced once for point cloud and voxels.  Move outside for efficency TODO
void CloudConstructor::calcPoints(bool freeResultPointsOnGPU) {

	//	size_t free, tot;
	//	cudaMemGetInfo(&free, &tot);	
	//	std::cout << "ENTER calcPoints::cudaMemGetInfo " << free << " " << tot << std::endl;

	if(isInitNeeded) {
		isInitNeeded = false;
		// assume all images have same dims 
		// TODO: is this safe?
		if(camCnt >= 1) {
			imgWidth = cams[0]->imgWidth;
			imgHeight = cams[0]->imgHeight;
			imgSize = imgWidth * imgHeight;
		} else {
			imgWidth = 0;
			imgHeight = 0;
			imgSize = 0;
		}


		std::cout << "image dims " << imgWidth << " x " << imgHeight << std::endl;
		pointCnt = imgWidth*imgHeight*camCnt + 1; //zeroth point is for junk!

		//assume params dont' change after first grab
		// TODO: is this safe?
		//		CamParams *ptParam = &CamParams[0];
		int paramI = 0;


		for(int i = 0; i < camCnt; i++) {

			camParams[paramI++]  = cams[i]->params.cx;
			camParams[paramI++]= cams[i]->params.cy;

			camParams[paramI++]= cams[i]->params.imageCenterU;
			camParams[paramI++]= cams[i]->params.imageCenterV;

			camParams[paramI++]= cams[i]->params.tx;
			camParams[paramI++]= cams[i]->params.ty;
			camParams[paramI++]= cams[i]->params.tz;

			camParams[paramI++]= cams[i]->params.rx;
			camParams[paramI++]= cams[i]->params.ry;
			camParams[paramI++]= cams[i]->params.rz;
		}

		paramI = 0;
		for(int i = 0; i < camCnt; i++) {
			std::cout  << i << " cx " << camParams[paramI++]  << std::endl;
			std::cout  << i << " cy " << camParams[paramI++]  << std::endl;

			std::cout  << i << " cu " << camParams[paramI++]  << std::endl;
			std::cout  << i << " cv " << camParams[paramI++]  << std::endl;

			std::cout  << i << " tx " << camParams[paramI++]  << std::endl;
			std::cout  << i << " ty " << camParams[paramI++]  << std::endl;
			std::cout  << i << " tz " << camParams[paramI++]  << std::endl;

			std::cout  << i << " rx " << camParams[paramI++]  << std::endl;
			std::cout  << i << " ry " << camParams[paramI++]  << std::endl;
			std::cout  << i << " rz " << camParams[paramI++]  << std::endl;

		}


		points = (float *)malloc(pointCnt * 3 * sizeof(float));
		//		points = new float[pointCnt * 3]; 
		if(points==NULL) {
			std::cerr << "CloudConstructor::calcPoints unable to allocate memory for " << pointCnt << " points" << std::endl;
			exit(1);
		}
		memset(points, 0, pointCnt * 3 * sizeof(float));
	}


#ifdef USE_CPU
	// extra allocation and de-allocation is done here to stay consistant with GPU code
	size_t paramSize = sizeof(double) * camCnt * CAM_PARAM_CNT;
	double *d_params;
	d_params = (double*) malloc(paramSize); // need to look up safecall TODO
	memcpy(d_params, camParams, paramSize);

	size_t transformSize = sizeof(double) * camCnt * 12;
	double *d_tansforms;
	d_tansforms =(double*) malloc(transformSize); // need to look up safecall TODO
	double* transPrt =d_tansforms;

	size_t singleTransformSize = sizeof(double) * 12;	
	for(int i = 0; i < camCnt; i++) {
		memcpy(transPrt, cams[i]->tMatrix, singleTransformSize);
		transPrt+=12;
	}

	size_t singleZImageArrSize = imgSize * sizeof(unsigned short);
	unsigned short *d_zimg;
	d_zimg = (unsigned short*) malloc(singleZImageArrSize * camCnt); // need to look up safecall TODO

	for(int i = 0; i < camCnt;i++) {
		memcpy(&d_zimg[i*imgSize], cams[i]->getZImage(), singleZImageArrSize);
	}


	size_t resultSize = pointCnt * 3 * sizeof(float);
	d_resultPoints = (float*) malloc(resultSize);

	cpu_calcPointCloud(camCnt,imgWidth,imgHeight, d_params, d_tansforms, d_zimg, d_resultPoints);


	//	std::cout << "done caluclated doing mem copy for " << resultSize << "points"; 

	memcpy(points, d_resultPoints, resultSize);

	//	std::cout << "done with mem copy"; 


	free(d_params);
	free(d_tansforms);
	free(d_zimg);
	free(d_resultPoints);
	d_resultPoints = NULL; // does free set to null for you?


#else // use gpu
	size_t paramSize = sizeof(double) * camCnt * CAM_PARAM_CNT;
	double *d_params;
	cutilSafeCall( cudaMalloc((void**)&d_params, paramSize) ); // need to look up safecall TODO
	cutilSafeCall( cudaMemcpy(d_params, camParams, paramSize, cudaMemcpyHostToDevice) );



	size_t transformSize = sizeof(double) * camCnt * 12;
	//	size_t transformSize = sizeof(double) * camCnt * 16;
	double *d_tansforms;
	cutilSafeCall( cudaMalloc((void**)&d_tansforms, transformSize) ); // need to look up safecall TODO

	double* transPrt =d_tansforms;
	//	size_t singleTransformSize = sizeof(double) * 16;	
	size_t singleTransformSize = sizeof(double) * 12;	
	for(int i = 0; i < camCnt; i++) {
		cutilSafeCall( cudaMemcpy(transPrt, cams[i]->tMatrix, singleTransformSize, cudaMemcpyHostToDevice) );
		transPrt+=12;
	}


	size_t singleZImageArrSize = imgSize * sizeof(unsigned short);
	unsigned short *d_zimg;
	cutilSafeCall( cudaMalloc((void**)&d_zimg, singleZImageArrSize * camCnt) ); // need to look up safecall TODO

	for(int i = 0; i < camCnt;i++) {
		cutilSafeCall( cudaMemcpy(&d_zimg[i*imgSize], cams[i]->getZImage(), singleZImageArrSize, cudaMemcpyHostToDevice) );
	}


	size_t resultSize = pointCnt * 3 * sizeof(float);
	cutilSafeCall( cudaMalloc((void**)&d_resultPoints, resultSize));

	gpu_calcPointCloud(camCnt,imgWidth,imgHeight, d_params, d_tansforms, d_zimg, d_resultPoints);


	//	std::cout << "done caluclated doing mem copy for " << resultSize << "points"; 

	cutilSafeCall(cudaMemcpy(points, d_resultPoints, resultSize, cudaMemcpyDeviceToHost));

	//	std::cout << "done with mem copy"; 


	cudaFree(d_params);
	cudaFree(d_tansforms);
	cudaFree(d_zimg);
	if(freeResultPointsOnGPU)
		freeGPUPoints();
#endif // USE_CPU

	//	cudaFree(d_resultPoints);


	//	cudaMemGetInfo(&free, &tot);	
	//	std::cout << "EXIT  calcPoints::cudaMemGetInfo " << free << " " << tot << std::endl;

}

//void CloudConstructor::getBounds(Vec3f  &min, Vec3f  &max) {
//}

CloudConstructor::~CloudConstructor() {
	delete camParams;
	delete points;
}
void CloudConstructor::cullCylinder(float cx, float cz, float r, float ceilingHack) {
	float rSqr = r*r;
	int j;
	float x,y,z,dx, dz;
	for(int i = 0; i < pointCnt; i++) {
		 j = i*3;
		 x = points[j];
		 y = points[j+1];
		 z = points[j+2];
		 dx = cx - x;
		 dx *=dx;
		 dz = cz - z;
		 dz *= dz;
		 dx+=dz;
		 if((dx < rSqr) || (y > ceilingHack)) {
			 points[j++] =0;
			points[j++] =0;
			points[j] =0;
		 }
	}
		 

}

void CloudConstructor::cull(float ax, float az, float bx, float bz, float floor) {


	// these should be z's not ys (confusing var names)
	float axbx = bx-ax;
	float byay = bz-az;
	float r = axbx*byay;

	for(int i = 0; i < pointCnt; i++) {
		int j = i*3;
		float cx = points[j];
		float cy = points[j+1];
		float cz = points[j+2];


		if((cx>ax) && (cx<bx)) {
			float r1 = axbx * (cz-az);
			float r2 = (bx-cx) * byay;
			if((cy<floor) || (r-r1-r2 >0)) {
				points[j++] =0;
				points[j++] =0;
				points[j] =0;
			}
		}
	}
}


void CloudConstructor::cpu_calcPointCloud(int camCnt, int imgWidth, int imgHeight, double* params, double* transforms, unsigned short *imgs, float* results) 
{
	for(int t = 0; t < camCnt; t++) {
		for(int u = 0; u < imgWidth; u++) {
			for(int v = 0; v < imgHeight; v++) {
				bool isValidIndex = (u >= 0) && (v >= 0) && (t>=0);
				int pixelIndex = isValidIndex ? ( (t * imgWidth * imgHeight) + (v * imgWidth) + u ) : -1; 
				unsigned short z = isValidIndex ? imgs[pixelIndex] : 0;	
				isValidIndex = (z != 0) && isValidIndex;


				double* paramPt = &params[t * CAM_PARAM_CNT];
				double cx = *paramPt++;
				double cy = *paramPt++;
				double cu = *paramPt++;
				double cv = *paramPt;	


				double x = isValidIndex ?  ((u - cu) *  cx * (double) z) : 0.0;
				double y = isValidIndex ?  (-(v - cv) *  cy * (double) z) : 0.0;


				pixelIndex = isValidIndex ? pixelIndex++ : 0 ;
				pixelIndex *= 3;


				double *m = &transforms[t*12];
				double tx =  m[0] * x	+  m[1] * y	+ m[2] * ((double) z) + m[9];	
				double ty =  m[3] * x	+  m[4] * y	+ m[5] * ((double) z) + m[10];	
				double tz =  m[6] * x	+  m[7] * y	+ m[8] * ((double) z) + m[11];

//				double tx =  m[0] * x	+  m[3] * y	+ m[6] * ((double) z) + m[9];	
//				double ty =  m[1] * x	+  m[4] * y	+ m[7] * ((double) z) + m[10];	
//				double tz =  m[2] * x	+  m[5] * y	+ m[8] * ((double) z) + m[11];

				results[pixelIndex++]	=  isValidIndex ? (float) tx : 0.0f ;
				results[pixelIndex++]	=  isValidIndex ? (float) ty : 0.0f ; 
				results[pixelIndex]		=  isValidIndex ? (float) tz : 0.0f ;
			}
		}
	}		
}