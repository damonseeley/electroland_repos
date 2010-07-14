#include "VoxelRenderer.h"
#include "math.h"




VoxelRenderer::VoxelRenderer(Voxel *voxel) {
	setVoxel(voxel);
	constructDisplayList();
	from = 0;
	to = voxel->divisions.z;
}

void VoxelRenderer::constructDisplayList() {
	Vec3f sides = (voxel->maxDim-voxel->minDim);
	sides /= voxel->divisions * 2.0f;

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
	float p = ((float)slice) / voxel->divisions.z;
	Vec3f bf = colors[FRONT][FRONT] * p + colors[BACK][FRONT] * (1-p);
	Vec3f bbk = colors[FRONT][BACK] * p + colors[BACK][BACK] * (1-p);
	Vec3f bl = colors[FRONT][LEFT] * p + colors[BACK][LEFT] * (1-p);
	Vec3f br = colors[FRONT][RIGHT] * p + colors[BACK][RIGHT] * (1-p);
	Vec3f bt = colors[FRONT][TOP] * p + colors[BACK][TOP] * (1-p);
	Vec3f bbt = colors[FRONT][BOT] * p + colors[BACK][BOT] * (1-p);

	 p = ((float)(slice+1)) / voxel->divisions.z;
	Vec3f ff = colors[FRONT][FRONT] * p + colors[BACK][FRONT] * (1-p);
	Vec3f fbk = colors[FRONT][BACK] * p + colors[BACK][BACK] * (1-p);
	Vec3f fl = colors[FRONT][LEFT] * p + colors[BACK][LEFT] * (1-p);
	Vec3f fr = colors[FRONT][RIGHT] * p + colors[BACK][RIGHT] * (1-p);
	Vec3f ft = colors[FRONT][TOP] * p + colors[BACK][TOP] * (1-p);
	Vec3f fbt = colors[FRONT][BOT] * p + colors[BACK][BOT] * (1-p);
	
	int i = 0;
	//top 
	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z; 
	colorVals[i++] = bt.x; colorVals[i++] = bt.y; colorVals[i++] = bt.z;
	colorVals[i++] = ft.x; colorVals[i++] = ft.y; colorVals[i++] = ft.z;
	colorVals[i++] = ft.x; colorVals[i++] = ft.y; colorVals[i++] = ft.z;

	colorVals[i++] = fbt.x; colorVals[i++] = fbt.y; colorVals[i++] = fbt.z; 
	colorVals[i++] = fbt.x; colorVals[i++] = fbt.y; colorVals[i++] = fbt.z;
	colorVals[i++] = bbt.x; colorVals[i++] = bbt.y; colorVals[i++] = bbt.z;
	colorVals[i++] = bbt.x; colorVals[i++] = bbt.y; colorVals[i++] = bbt.z;

	colorVals[i++] = ff.x; colorVals[i++] = ff.y; colorVals[i++] = ff.z; 
	colorVals[i++] = ff.x; colorVals[i++] = ff.y; colorVals[i++] = ff.z;
	colorVals[i++] = ff.x; colorVals[i++] = ff.y; colorVals[i++] = ff.z;
	colorVals[i++] = ff.x; colorVals[i++] = ff.y; colorVals[i++] = ff.z;

	colorVals[i++] = bbk.x; colorVals[i++] = bbk.y; colorVals[i++] = bbk.z; 
	colorVals[i++] = bbk.x; colorVals[i++] = bbk.y; colorVals[i++] = bbk.z;
	colorVals[i++] = bbk.x; colorVals[i++] = bbk.y; colorVals[i++] = bbk.z;
	colorVals[i++] = bbk.x; colorVals[i++] = bbk.y; colorVals[i++] = bbk.z;

	colorVals[i++] = fl.x; colorVals[i++] = fl.y; colorVals[i++] = fl.z; 
	colorVals[i++] = bl.x; colorVals[i++] = bl.y; colorVals[i++] = bl.z;
	colorVals[i++] = bl.x; colorVals[i++] = bl.y; colorVals[i++] = bl.z;
	colorVals[i++] = fl.x; colorVals[i++] = fl.y; colorVals[i++] = fl.z;

	colorVals[i++] = br.x; colorVals[i++] = br.y; colorVals[i++] = br.z; 
	colorVals[i++] = fr.x; colorVals[i++] = fr.y; colorVals[i++] = fr.z;
	colorVals[i++] = fr.x; colorVals[i++] = fr.y; colorVals[i++] = fr.z;
	colorVals[i++] = br.x; colorVals[i++] = br.y; colorVals[i++] = br.z;
}


void VoxelRenderer::setVoxel(Voxel *voxel) {
	this->voxel = voxel;
}

void VoxelRenderer::setFromTo(float from, float to) {
	Vec3f size = (voxel->maxDim-voxel->minDim);
	Vec3f sides = size / voxel->divisions;
	this->from = voxel->divisions.z *  ((from - voxel->minDim.z) / size.z) ;
	this->to = voxel->divisions.z *((to - voxel->minDim.z) / size.z );
}
void VoxelRenderer::draw(DWORD curTime, float dt, float renderThresh) {

	glEnableClientState(GL_VERTEX_ARRAY);
	glEnableClientState(GL_COLOR_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, cubePoints);


	Vec3f size = (voxel->maxDim-voxel->minDim);
	Vec3f sides = size / voxel->divisions;

	float xTrans = 0;
	float yTrans = 0;

//	int kFrom = voxel->divisions.z *  ((from - voxel->minDim.z) / size.z) ;
//	int kTo = voxel->divisions.z *((to - voxel->minDim.z) / size.z );


	float zTrans = voxel->minDim.z + (sides.z * .5f) + (sides.z * from);

	float* gridPtr = &voxel->grid[from * voxel->divisions.x * voxel->divisions.y];

	for(int k = from; k < to; k++) {
		zTrans += sides.z;
		yTrans = voxel->minDim.y + (sides.y * .5f);
		constructColorList(k);
		glColorPointer(3, GL_FLOAT, 0, colorVals);
		for(int j = 0; j < voxel->divisions.y; j++) {
			yTrans += sides.y;
			xTrans = voxel->minDim.x + (sides.x * .5f);
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

