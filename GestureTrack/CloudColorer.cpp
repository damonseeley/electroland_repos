#include "CloudColorer.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>


extern "C" void gpu_calcColor(int pointCnt, float* pixels, float minZ, float diffZ, float minC1, float minC2, float minC3, float diffC1, float diffC2, float diffC3, bool hsv, bool quads, float* colors);

CloudColorer::CloudColorer(float minZ, float maxZ, Vec3f minColor, Vec3f maxColor, bool hsv, float size) {
	this->minZ = minZ;
	diffZ = maxZ-minZ;
	this->minColor = minColor;
	this->diffColor = maxColor - minColor;
	this->hsv = hsv;
	this->size = size;
	colors = NULL;
	quads = NULL;
	drawQuads = false;
}

void CloudColorer::calcQuads(int pointCloudCnt,  float* pointCloud) {
	if(quads == NULL) {	//init once
		quads = (float *) malloc(pointCloudCnt * sizeof(float) * 12);

	}
	memset(quads,0,pointCloudCnt * sizeof(float) * 12);

	float halfSize = size*0.5f;

	float x,y,z;
	int j;
	float left;
	float right;
	float top;
	float bottom;

	for(int i = 0; i < (3* pointCloudCnt); i+=3) { // zerois is junk
		x = pointCloud[i];
		y = pointCloud[i+1];
		z = pointCloud[i+2];
		if(z != 0.0) {
		j = i*4;
		left = x-halfSize;
		right = x+halfSize;
		top = y+halfSize;
		bottom = y-halfSize;

		quads[j++] = left;
		quads[j++] = top;
		quads[j++] = z;

		quads[j++] = right;
		quads[j++] = top;
		quads[j++] = z;

		quads[j++] = right;
		quads[j++] = bottom;
		quads[j++] = z;
		

		quads[j++] = left;
		quads[j++] = bottom;
		quads[j++] = z;
		}


	}
}
void CloudColorer::calcColors(int pointCloudCnt,  float* pointCloud,   bool cloudIsOnGPU){
	float *d_pointCloud;
	float *d_colors;

	size_t pntCloudSize = sizeof(float) * pointCloudCnt * 3;
	size_t colorArSize = pntCloudSize;
	if(drawQuads) {
		colorArSize *=4;
	} 

	if(colors == NULL) {		//init once
		colors = (float *) malloc(colorArSize);

	}

/*
		
	for(int ix = 0; ix < pointCloudCnt; ix++) {
		int pointCnt = pointCloudCnt;
		float* pixels = pointCloud;
		float minC1 = minColor.x;
float minC2 = minColor.y;
float minC3 = minColor.z;
float diffC1 = diffColor.x;
float diffC2 = diffColor.y;
float diffC3 = diffColor.z;


	bool validPoint = ((ix > 0) && (ix < pointCnt))? true : false;
	int i = ix * 3;	
	float z = validPoint ? pixels[i+2] : 0.0;
	if(z != 0.0) { // only equal to 0.0 if set as invalid in cloud contructor	
		float distZ = z - minZ;
		float percent = distZ/diffZ;
		percent = percent<0.0 ? 0.0 : percent;
		percent = percent>1.0 ? 1.0 : percent;
	
		float r; 	
		float g; 
		float b;
		if(hsv) {
//			HSV2RGB(minC1 + percent * diffC1, minC2 + percent * diffC2, minC3 + percent * diffC3, r,g,b); 		
		} else {
			r= minC1 + percent * diffC1;
			g= minC2 + percent * diffC2;
			b= minC3 + percent * diffC3;
		}
		
		if(quads) {
			i*=4;
			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i++] = b;

			colors[i++] = r;
			colors[i++] = g;
			colors[i] = b;
			
		} else {
			colors[i++] = r;
			colors[i++] = g;
			colors[i] = b;
		}



	}
	}

*/
	if(! cloudIsOnGPU) {
	cutilSafeCall( cudaMalloc((void**)&d_pointCloud, pntCloudSize) ); // need to look up safecall TODO
	cutilSafeCall( cudaMemcpy(d_pointCloud, pointCloud, pntCloudSize, cudaMemcpyHostToDevice) );
	} else {
	d_pointCloud = pointCloud;
	}

	cutilSafeCall( cudaMalloc((void**)&d_colors, colorArSize) ); // need to look up safecall TODO
	cudaMemset(d_colors, 0, colorArSize);


	gpu_calcColor( pointCloudCnt, d_pointCloud,	minZ, diffZ, minColor.x, minColor.y, minColor.z, diffColor.x, diffColor.y, diffColor.z, hsv, drawQuads, d_colors);

	cutilSafeCall( cudaMemcpy(colors, d_colors, colorArSize, cudaMemcpyDeviceToHost) );

	cudaFree(d_colors);
	


}