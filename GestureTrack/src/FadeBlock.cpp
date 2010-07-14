#include "FadeBlock.h"
#include <stdio.h>
#include <iostream>


GLuint  FadeBlock::displayList = 0;

FadeBlock::FadeBlock(DWORD startTime, DWORD endTime, float x, float y, float z, float r,float g,float b) {
	this->startTime = startTime;
	this->endTime = endTime;
	this->x = x;
	this->y = y;
	this->z = z;
	this->r = r;
	this->g = g;
	this->b = b;
	timeScale = 1.0f / (float) (endTime - startTime);
}

void FadeBlock::createDisplayList(Vec3f sides) {

	sides /= 2.0;
	FadeBlock::displayList = glGenLists(1);
	glNewList(FadeBlock::displayList,GL_COMPILE);

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

bool FadeBlock::draw(DWORD curTime) {
	if(curTime > endTime) {
		return false;
	} else {
		float alpha = endTime-curTime;
		alpha *= timeScale;
//		std::cout << endTime << " - " << curTime << " " << alpha << std::endl;;
glEnable (GL_BLEND);
glColor4f(r,g,b,alpha);
		glPushMatrix();
		glTranslatef(x,y,z); // add .5 boxes are centered on grid
		glCallList(displayList);
		glPopMatrix();
		return true;
	}

}

