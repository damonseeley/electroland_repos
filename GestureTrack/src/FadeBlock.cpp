#include "FadeBlock.h"
#include <stdio.h>
#include <iostream>


GLuint  FadeBlock::displayList = 0;

FadeBlock::FadeBlock(DWORD startTime, DWORD endTime, float x, float y, float z) {
	this->startTime = startTime;
	this->endTime = endTime;
	this->x = x;
	this->y = y;
	this->z = z;
	timeScale = 1.0f / (float) (endTime - startTime);
}

void FadeBlock::createDisplayList(Vec3f sides) {

	sides /= 2;
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
		float alpha = endTime - curTime;
		alpha *= timeScale;
		std::cout << alpha << std::endl;;
		glColor4f(0.1f, 0.1f, alpha ,alpha);
		glPushMatrix();
		glTranslatef(x,y,z); // add .5 boxes are centered on grid
		glCallList(displayList);
		glPopMatrix();
		return true;
	}

}

