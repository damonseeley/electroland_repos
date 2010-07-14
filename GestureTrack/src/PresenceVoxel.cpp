#include "PresenceVoxel.h"


PresenceVoxel::PresenceVoxel(Vec3f minDim, Vec3f maxDim, Vec3i divisions, bool createDL) : Voxel(minDim, maxDim, divisions, false) {
	if(createDL) {
		createDisplayList();
//		constructFloorPoints();
	}

}	


void PresenceVoxel::createDisplayList() {

	Vec3f sides = (maxDim-minDim);
	sides /= divisions * 2;
	displayList = glGenLists(1);
	glNewList(displayList,GL_COMPILE);

	glBegin(GL_QUADS);
	glVertex3f(sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Top)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Top)
	glVertex3f(-sides.x, sides.y, sides.z);  // Bottom Left Of The Quad (Top)
	glVertex3f(sides.x, sides.y, sides.z);  // Bottom Right Of The Quad (Top)

	glVertex3f(sides.x, -sides.y, sides.z);  // Top Right Of The Quad (Bottom)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Top Left Of The Quad (Bottom)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Left Of The Quad (Bottom)
	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Right Of The Quad (Bottom)

	glVertex3f(sides.x, sides.y, sides.z);  // Top Right Of The Quad (Front)
	glVertex3f(-sides.x, sides.y, sides.z);  // Top Left Of The Quad (Front)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Bottom Left Of The Quad (Front)
	glVertex3f(sides.x, -sides.y, sides.z);  // Bottom Right Of The Quad (Front)

	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Left Of The Quad (Back)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Right Of The Quad (Back)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Back)
	glVertex3f(sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Back)

	glVertex3f(-sides.x, sides.y, sides.z);  // Top Right Of The Quad (Left)
	glVertex3f(-sides.x, sides.y, -sides.z);  // Top Left Of The Quad (Left)
	glVertex3f(-sides.x, -sides.y, -sides.z); // Bottom Left Of The Quad (Left)
	glVertex3f(-sides.x, -sides.y, sides.z);  // Bottom Right Of The Quad (Left)

	glVertex3f(sides.x, sides.y, -sides.z);  // Top Right Of The Quad (Right)
	glVertex3f(sides.x, sides.y, sides.z);  // Top Left Of The Quad (Right)
	glVertex3f(sides.x, -sides.y, sides.z);  // Bottom Left Of The Quad (Right)
	glVertex3f(sides.x, -sides.y, -sides.z);  // Bottom Right Of The Quad (Right)
	// Done Drawing The Cube
	glEnd();
	glEndList();


}
void PresenceVoxel::draw(float renderThresh) {
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
			float alpha = 1.0f - ((float) j / (float) divisions.y);
			for(int i = 0; i < divisions.x; i++) {
				if(*gridPtr > renderThresh) {
				
					float val = *gridPtr / 50.0f;
					val = (val > 1.0)? 1.0 : val;
					glColor4f(val,val,1.0f ,alpha);
					glPushMatrix();
					glTranslatef((i+.5) * sides.x, (j+.5) * sides.y, (k+.5) * sides.z); // add .5 boxes are centered on grid
					glCallList(displayList);
					glPopMatrix();
				}
				gridPtr++;
			}
		}
	}

}
