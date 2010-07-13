#include "VoxelRenderer.h"



VoxelRenderer::VoxelRenderer(Voxel *voxel) {
	setVoxel(voxel);
	constructDisplayList();
}

void VoxelRenderer::constructDisplayList() {
	Vec3f sides = (voxel->maxDim-voxel->minDim);
	sides /= voxel->divisions * 2.0;

	int i = 0;
	//top
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	//bot
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;
	//front
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	//back
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	//left
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = -sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	//right
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = -sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = sides.z;
	cubePoints[i++] = sides.x;	cubePoints[i++] = -sides.y;	cubePoints[i++] = -sides.z;

}

void VoxelRenderer::setFrontColor(Vec3f f, Vec3f back, Vec3f l, Vec3f r, Vec3f t, Vec3f bot) {
	colors[FRONT][FRONT] = f;
	colors[FRONT][BACK] = back;
	colors[FRONT][LEFT] = l;
	colors[FRONT][RIGHT] = r;
	colors[FRONT][TOP] = t;
	colors[FRONT][BOT] = bot;
}
void VoxelRenderer::setBackColor (Vec3f f, Vec3f back, Vec3f l, Vec3f r, Vec3f t, Vec3f bot) {
	colors[BACK][FRONT] = f;
	colors[BACK][BACK] = back;
	colors[BACK][LEFT] = l;
	colors[BACK][RIGHT] = r;
	colors[BACK][TOP] = t;
	colors[BACK][BOT] = bot;
}

void VoxelRenderer::constructColorList(int slice) {
	float p = slice / voxel->divisions.z;
	Vec3f f = colors[FRONT][FRONT] * p + colors[BACK][FRONT] * (1-p);
	Vec3f bk = colors[FRONT][BACK] * p + colors[BACK][BACK] * (1-p);
	Vec3f l = colors[FRONT][LEFT] * p + colors[BACK][LEFT] * (1-p);
	Vec3f r = colors[FRONT][RIGHT] * p + colors[BACK][RIGHT] * (1-p);
	Vec3f t = colors[FRONT][TOP] * p + colors[BACK][TOP] * (1-p);
	Vec3f bt = colors[FRONT][BOT] * p + colors[BACK][BOT] * (1-p);
	
	int i = 0;
	colorVals[i++] = t.x; colorVals[i++] = t.y; colorVals[i++] = t.z; 
	colorVals[i++] = t.x; colorVals[i++] = t.y; colorVals[i++] = t.z;
	colorVals[i++] = t.x; colorVals[i++] = t.y; colorVals[i++] = t.z;
	colorVals[i++] = t.x; colorVals[i++] = t.y; colorVals[i++] = t.z;

	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z; 
	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z;
	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z;
	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z;

	colorVals[i++] = f.x; colorVals[i++] = f.y; colorVals[i++] = f.z; 
	colorVals[i++] = f.x; colorVals[i++] = f.y; colorVals[i++] = f.z;
	colorVals[i++] = f.x; colorVals[i++] = f.y; colorVals[i++] = f.z;
	colorVals[i++] = f.x; colorVals[i++] = f.y; colorVals[i++] = f.z;

	colorVals[i++] = bk.x; colorVals[i++] = bk.y; colorVals[i++] = bk.z; 
	colorVals[i++] = bk.x; colorVals[i++] = bk.y; colorVals[i++] = bk.z;
	colorVals[i++] = bk.x; colorVals[i++] = bk.y; colorVals[i++] = bk.z;
	colorVals[i++] = bk.x; colorVals[i++] = bk.y; colorVals[i++] = bk.z;

	colorVals[i++] = l.x; colorVals[i++] = l.y; colorVals[i++] = l.z; 
	colorVals[i++] = l.x; colorVals[i++] = l.y; colorVals[i++] = l.z;
	colorVals[i++] = l.x; colorVals[i++] = l.y; colorVals[i++] = l.z;
	colorVals[i++] = l.x; colorVals[i++] = l.y; colorVals[i++] = l.z;

	colorVals[i++] = r.x; colorVals[i++] = r.y; colorVals[i++] = r.z; 
	colorVals[i++] = r.x; colorVals[i++] = r.y; colorVals[i++] = r.z;
	colorVals[i++] = r.x; colorVals[i++] = r.y; colorVals[i++] = r.z;
	colorVals[i++] = r.x; colorVals[i++] = r.y; colorVals[i++] = r.z;
}


void VoxelRenderer::setVoxel(Voxel *voxel) {
	this->voxel = voxel;
}

void VoxelRenderer::draw(DWORD curTime, float dt, float renderThresh) {

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, cubePoints);


	float* gridPtr = voxel->grid;
	Vec3f sides = (voxel->maxDim-voxel->minDim);
	sides /= voxel->divisions;

	float xTrans = 0;
	float yTrans = 0;
	float zTrans = -sides.z * .5f;
	for(int k = 0; k < voxel->divisions.z; k++) {
		zTrans += sides.z;
		yTrans = -sides.y * .5f;
		constructColorList(k);
		glColorPointer(3, GL_FLOAT, 0, colorVals);
		for(int j = 0; j < voxel->divisions.y; j++) {
			yTrans += sides.y;
			xTrans = -sides.x * .5f;
			for(int i = 0; i < voxel->divisions.x; i++) {
				xTrans += sides.x;
				if(*gridPtr > renderThresh) {
					glPushMatrix();
					glTranslatef(xTrans, yTrans, zTrans);
					glDrawArrays(GL_QUADS,0, 24);
					glPopMatrix();
				}
				gridPtr++;
			}
		}
	}
	glDisableClientState(GL_COLOR_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);



}
