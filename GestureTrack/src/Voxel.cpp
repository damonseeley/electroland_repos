#include <iostream>
#include <sstream>

#include "Voxel.h"

#include <cuda_runtime.h>
#include <cutil_inline.h>

//#include<voxel.cu>

extern "C" void gpu_calcVoxel(int pointCnt, float* pixels, float minX, float minY, float minZ, float maxX, float maxY, float maxZ, int divX, int divY, int divZ, float* voxGrid);
extern "C" void gpu_addScale(int gridSize, float* d_this, float a, float* d_that, float b);
extern "C" void gpu_thresh(int gridSize, float* d_this, float thresh);
extern "C" void gpu_threshSet(int gridSize, float* d_this, float thresh, float belowVal, float aboveVal);
extern "C" void gpu_scalarMult(int gridSize, float* d_this, float val);

extern "C" void gpu_mult(int gridSize, float* d_this, float* d_that);
extern "C" void gpu_add(int gridSize, float* d_this, float* d_that);
extern "C" void gpu_sub(int gridSize, float* d_this, float* d_that);
extern "C" void gpu_sub2(int gridSize, float* d_this, float* a, float* b);
extern "C" void gpu_incIfOverThresh(int gridSize, float *d_this, float* d_that, float thresh);
extern "C" void gpu_scaleDownFrom(int gridSize, int dx, int dy, int dz, float *d_this, float *d_that);


//float* Voxel::glFloorPoints= NULL;
//float* Voxel::glFloorColors = NULL;
//float Voxel::floorGridPointCnt = 0.0f;
GLuint Voxel::displayList = 0;

Voxel::Voxel(Vec3f minDim, Vec3f maxDim, Vec3i divisions, bool createDL) {
	this->minDim = minDim;
	this->maxDim = maxDim;
	this->divisions = divisions;
	gridSize = divisions.x * divisions.y * divisions.z;
	voxMemSize = sizeof(float) * gridSize;
	grid = (float *) malloc(voxMemSize);
	memset(grid,0,voxMemSize);

	//		grid = new int[gridSize];
	if(grid == NULL) {
		std::cerr << "Voxel cannot allocate memory for grid of size " << divisions;
		exit(1);
	}
	if(createDL) {
//		if(glFloorPoints == NULL) {
			createDisplayList();
//			constructFloorPoints();
		//}
	}
	d_vox = NULL;



}

Voxel::~Voxel() {
	free(grid);
}
/*
void Voxel::constructFloorPoints() {
	Voxel::floorGridPointCnt = (2 * (divisions.x +1)) + (2 * (divisions.y+1)) + (2 * (divisions.x +1)) +(2 * (divisions.z +1));
	Voxel::glFloorColors = new float[floorGridPointCnt * 3];
	Voxel::glFloorPoints = new float[floorGridPointCnt * 3];

	float inc = (maxDim.y - minDim.y) / divisions.y;
	float cInc = .95f/divisions.y;
	float cur = minDim.y;
	float cCur = 0.0;
	int start = 0;
	int stop = 3*2*(divisions.y + 1);
	//back
	for(int i =start; i <= stop; i++) {
		glFloorColors[i] = cCur;
		glFloorPoints[i++] = minDim.x; 
		glFloorColors[i] = cCur;
		glFloorPoints[i++] = cur;
		glFloorColors[i] = cCur;
		glFloorPoints[i++] = minDim.z;

		glFloorColors[i] = cCur;
		glFloorPoints[i++] = maxDim.x; 
		glFloorColors[i] = cCur;
		glFloorPoints[i++] = cur;
		glFloorColors[i] = cCur;
		glFloorPoints[i] = minDim.z;
		cur+= inc;
		cCur += cInc;
	}

	inc = (maxDim.x - minDim.x) / divisions.x;
	cur = minDim.x;
	start = stop;
	stop+=3*2*(divisions.x + 1);
	for(int i =start; i <= stop; i++) {
		glFloorColors[i] = 0.0f;
		glFloorPoints[i++] = cur; 
		glFloorColors[i] = 0.0f;
		glFloorPoints[i++] = minDim.y;
		glFloorColors[i] = 0.0f;
		glFloorPoints[i++] = minDim.z;

		glFloorColors[i] = 1.0f;
		glFloorPoints[i++] = cur; 
		glFloorColors[i] = 1.0f;
		glFloorPoints[i++] = maxDim.y;
		glFloorColors[i] = 1.0f;
		glFloorPoints[i] = minDim.z;
		cur+= inc;
	}
	/*
	// floor
	start = stop;
	stop+=3*2*(divisions.z + 1);
	inc = (maxDim.z - minDim.z) / divisions.z;
	cur = minDim.z;

	for(int i =start; i <= stop; i++) {
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = minDim.x; 
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = maxDim.y;
		glFloorColors[i] =.05f;
		glFloorPoints[i++] = cur;

		glFloorColors[i] = .05f;
		glFloorPoints[i++] = maxDim.x; 
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = maxDim.y;
		glFloorColors[i] = cCur;
		glFloorPoints[i] = cur;
		cur+= inc;
	}
	inc = (maxDim.x - minDim.x) / divisions.x;
	cur = minDim.x;
	start = stop;
	stop+=3*2*(divisions.x + 1);
	for(int i =start; i <= stop; i++) {
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = cur; 
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = maxDim.y;
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = minDim.z;

		glFloorColors[i] = .05f;
		glFloorPoints[i++] = cur; 
		glFloorColors[i] = .05f;
		glFloorPoints[i++] = maxDim.y;
		glFloorColors[i] = .05f;
		glFloorPoints[i] =  maxDim.z;
		cur+= inc;
	}
*/

//}


void Voxel::draw(float renderThresh) {
	/*
	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glColorPointer(3, GL_FLOAT, 0, glFloorColors);
	glVertexPointer(3, GL_FLOAT, 0, glFloorPoints);
	glDrawArrays(GL_LINES,0, floorGridPointCnt);
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);
*/
	Vec3f sides = (maxDim-minDim);
	sides /= divisions;

	float* gridPtr = grid;

	for(int k = 0; k < divisions.z; k++) {
		for(int j = 0; j < divisions.y; j++) {
			for(int i = 0; i < divisions.x; i++) {
				if(*gridPtr > renderThresh) {
					glPushMatrix();
					glTranslatef(minDim.x + (i+.5) * sides.x, minDim.y+ (j+.5) * sides.y, minDim.z + (k+.5) * sides.z); // add .5 boxes are centered on grid
					glCallList(displayList);
					glPopMatrix();
				}
				gridPtr++;
			}
		}
	}

}
void Voxel::copyGrid(Voxel *vox) {
		memcpy(grid, vox->grid, voxMemSize);
}

void Voxel::createDisplayList() {

	Vec3f sides = (maxDim-minDim);
	sides /= divisions * 2;
	displayList = glGenLists(1);
	glNewList(displayList,GL_COMPILE);

	glBegin(GL_QUADS);
	glColor3f(0.0f, 1.0f, 0.0f);  // Set The Color To Green
	glVertex3f(sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Top)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Top)
	glVertex3f(-sides.x, sides.y, sides.z);  // Bottom Left Of The Quad (Top)
	glVertex3f(sides.x, sides.y, sides.z);  // Bottom Right Of The Quad (Top)

	glColor3f( 1.0f, 0.5f, 0.0f);  // Set The Color To Orange
	glVertex3f(sides.x, -sides.y, sides.z);  // Top Right Of The Quad (Bottom)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Top Left Of The Quad (Bottom)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Left Of The Quad (Bottom)
	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Right Of The Quad (Bottom)

	glColor3f( 1.0f, 0.0f, 0.0f);  // Set The Color To Red
	glVertex3f(sides.x, sides.y, sides.z);  // Top Right Of The Quad (Front)
	glVertex3f(-sides.x, sides.y, sides.z);  // Top Left Of The Quad (Front)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Bottom Left Of The Quad (Front)
	glVertex3f(sides.x, -sides.y, sides.z);  // Bottom Right Of The Quad (Front)

	glColor3f( 1.0f,  1.0f, 0.0f);  // Set The Color To Yellow
	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Left Of The Quad (Back)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Right Of The Quad (Back)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Back)
	glVertex3f(sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Back)

	glColor3f(0.0f, 0.0f,  1.0f);  // Set The Color To Blue
	glVertex3f(-sides.x, sides.y, sides.z);  // Top Right Of The Quad (Left)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Left)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Left Of The Quad (Left)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Bottom Right Of The Quad (Left)

	glColor3f( 1.0f, 0.0f,  1.0f);  // Set The Color To Violet
	glVertex3f(sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Right)
	glVertex3f(sides.x, sides.y, sides.z);  // Top Left Of The Quad (Right)
	glVertex3f(sides.x, -sides.y, sides.z);  // Bottom Left Of The Quad (Right)
	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Right Of The Quad (Right)
	// Done Drawing The Cube
	glEnd();
	glEndList();


}

void Voxel::allocateGridOnGPU(bool copy) {
	if(d_vox == NULL) { // if not null already on device
		cutilSafeCall( cudaMalloc((void**)&d_vox, voxMemSize) ); // need to look up safecall TODO
		if(copy) {
				cutilSafeCall( cudaMemcpy(d_vox, grid, voxMemSize, cudaMemcpyHostToDevice) );
		} else {
				cudaMemset(d_vox, 0, voxMemSize);
		}
	}
}

void Voxel::deallocateGridOnGPU() {
		cudaFree(d_vox);
		d_vox = NULL;
}

void Voxel::calcVoxel(int pointCloudCnt, float* pointCloud, bool cloudIsOnGPU, bool freeGridFromGPU) {


	float *d_pointCloud;

	if(! cloudIsOnGPU) {
		size_t pntCloudSize = sizeof(float) * pointCloudCnt * 3;
		cutilSafeCall( cudaMalloc((void**)&d_pointCloud, pntCloudSize) ); // need to look up safecall TODO
		cutilSafeCall( cudaMemcpy(d_pointCloud, pointCloud, pntCloudSize, cudaMemcpyHostToDevice) );
	} else {
		d_pointCloud = pointCloud;
	}

	// use allocateGridOnGPU
//	cutilSafeCall( cudaMalloc((void**)&d_vox, voxMemSize) ); // need to look up safecall TODO
//	cudaMemset(d_vox, 0, voxMemSize);

	allocateGridOnGPU(false);

	gpu_calcVoxel( pointCloudCnt, d_pointCloud, minDim.x,minDim.y,minDim.z, maxDim.x, maxDim.y,maxDim.z, divisions.x, divisions.y, divisions.z, d_vox);

	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	if(freeGridFromGPU)
		deallocateGridOnGPU();
}


void Voxel::addScale(float a, Voxel *vox, float b, bool freeFromGPU) {
#ifdef _DEBUG
	if(this->divisions != vox->divisions) {
		std::cerr << "Unable to perform addsScale on voxels.  Divisions don't match" << std::endl;
	}
#endif


	allocateGridOnGPU();
	vox->allocateGridOnGPU();

	gpu_addScale(gridSize, d_vox, a, vox->d_vox, b);


	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	
	if(freeFromGPU) 
		deallocateGridOnGPU();


}


void Voxel::mult(Voxel *vox, bool freeFromGPU) {
#ifdef _DEBUG
	if(this->divisions != vox->divisions) {
		std::cerr << "Unable to perform addsScale on voxels.  Divisions don't match" << std::endl;
	}
#endif


	allocateGridOnGPU();
	vox->allocateGridOnGPU();

	gpu_mult(gridSize, d_vox, vox->d_vox);


	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	
	if(freeFromGPU) 
		deallocateGridOnGPU();


}

void Voxel::add(Voxel *vox, bool freeFromGPU) {
#ifdef _DEBUG
	if(this->divisions != vox->divisions) {
		std::cerr << "Unable to perform addsScale on voxels.  Divisions don't match" << std::endl;
	}
#endif


	allocateGridOnGPU();
	vox->allocateGridOnGPU();

	gpu_add(gridSize, d_vox, vox->d_vox);


	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	
	if(freeFromGPU) 
		deallocateGridOnGPU();


}



void Voxel::sub(Voxel *a, Voxel *b, bool freeFromGPU) {

	allocateGridOnGPU(false);
	a->allocateGridOnGPU();
	b->allocateGridOnGPU();

	gpu_sub2(gridSize, d_vox, a->d_vox, b->d_vox);


	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	
	if(freeFromGPU) 
		deallocateGridOnGPU();


}

void Voxel::sub(Voxel *vox, bool freeFromGPU) {
#ifdef _DEBUG
	if(this->divisions != vox->divisions) {
		std::cerr << "Unable to perform addsScale on voxels.  Divisions don't match" << std::endl;
	}
#endif


	allocateGridOnGPU();
	vox->allocateGridOnGPU();

	gpu_sub(gridSize, d_vox, vox->d_vox);


	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );

	
	if(freeFromGPU) 
		deallocateGridOnGPU();


}

void Voxel::thresh(float thresh, bool freeFromGPU) {
	allocateGridOnGPU();
	gpu_thresh(gridSize, d_vox, thresh);
	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );
	if(freeFromGPU) 
		deallocateGridOnGPU();
}


void Voxel::scalarMult(float val, bool freeFromGPU) {
	allocateGridOnGPU();
	gpu_scalarMult(gridSize, d_vox, val);
	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );
	if(freeFromGPU) 
		deallocateGridOnGPU();
}



void Voxel::threshSet(float thresh, float below, float above, bool freeFromGPU) {
	allocateGridOnGPU();
	gpu_threshSet(gridSize, d_vox, thresh, below, above);
	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );
	if(freeFromGPU) 
		deallocateGridOnGPU();
}

void Voxel::incIfOverThresh(Voxel *v, float thresh, bool freeFromGPU) {
	allocateGridOnGPU();
	v->allocateGridOnGPU();
	gpu_incIfOverThresh(gridSize, d_vox, v->d_vox, thresh);
	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );
	if(freeFromGPU) 
		deallocateGridOnGPU();

}

void Voxel::scaleDownFrom(Voxel *doubleDim, bool freeFromGPU) {
	
	allocateGridOnGPU(false); // just zero out
	doubleDim->allocateGridOnGPU();
	gpu_scaleDownFrom(gridSize, divisions.x, divisions.y, divisions.z, d_vox, doubleDim->d_vox);
	cutilSafeCall( cudaMemcpy(grid, d_vox, voxMemSize, cudaMemcpyDeviceToHost) );
	if(freeFromGPU) 
		deallocateGridOnGPU();

}

void Voxel:: scaleDownFrom_kernel(int gridSize, int dx, int dy, int dz, float *d_this, float *d_that) {

	for(int i = 0; i < gridSize; i++) {
		int z = i / (dx * dy);
		int r = i % (dx * dy);
		int y = r / (dx);
		int x = r % (dx);

		x*=2;
		y*=2;
		z*=2;

		d_this[i] = d_that[x + (y * (2 * dx)) + (z * (4 * dx * dy))];
		d_this[i] = d_that[x+1 + (y * (2 * dx)) + (z * (4 * dx * dy))];
		d_this[i] = d_that[x + ( (y+1) * (2 * dx)) + (z * (4 * dx * dy))];
		d_this[i] = d_that[x+1 + ((y+1) * (2 * dx)) + (z * (4 * dx * dy))];
		d_this[i] = d_that[x + (y * (2 * dx)) + ((z+1) * (4 * dx * dy))];
		d_this[i] = d_that[x+1 + (y * (2 * dx)) + ((z+1) * (4 * dx * dy))];
		d_this[i] = d_that[x + ( (y+1) * (2 * dx)) + ((z+1) * (4 * dx * dy))];
		d_this[i] = d_that[x+1 + ((y+1) * (2 * dx)) + ((z+1) * (4 * dx * dy))];

	}



}