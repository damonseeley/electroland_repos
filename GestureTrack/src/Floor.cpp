#include "Floor.h"

#include <GL/glut.h>


Floor::Floor(float level, float minX, float maxX, float depth, Vec3f backColor, Vec3f frontColor, float divX, float divZ) {
	this->level = level;
	this->minX = minX;
	this->maxX = maxX;
	this->depth = depth;
	this->backColor = backColor;
	this->frontColor = frontColor;
	this->divX = divX;
	this->divZ = divZ;
	this->stepX = (maxX-minX) / divX;
	this->stepZ = (0+depth) / divZ;
	this->stepC = (frontColor-backColor)/divZ;
}

void Floor::render() {
	glBegin(GL_LINES);
	for(float x = minX; x <= maxX; x+=stepX) 
	{
		glColor3f(frontColor.x, frontColor.y, frontColor.z);
		glVertex3f(x, level, 0);
		glColor3f(backColor.x, backColor.y, backColor.z);
		glVertex3f(x, level, depth);

	}
	Vec3f c = backColor;
	for(float z = depth; z <= 0;z+=stepZ) 
	{
		glColor3f(c.x,c.y, c.z);
		glVertex3f(minX, level, z);
		glVertex3f(maxX, level, z);
		c+=stepC;
	}
	glEnd();
}

