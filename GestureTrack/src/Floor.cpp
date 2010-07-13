#include "Floor.h"

#include <GL/glut.h>


Floor::Floor(float level, float minX, float maxX, float depth, Vec3f backColor, Vec3f frontColor) {
	this->level = level;
	this->minX = minX;
	this->maxX = maxX;
	this->depth = -depth;
	this->backColor = backColor;
	this->frontColor = frontColor;
}

void Floor::render() {
	glColor3f(frontColor.x, frontColor.y, frontColor.z);
	glBegin(GL_QUADS);
	glVertex3f(minX,level,0);
	glVertex3f(maxX,level,0);
	glColor3f(backColor.x, backColor.y, backColor.z);
	glVertex3f(maxX,level,depth);
	glVertex3f(minX,level,depth);
	glEnd();
}

