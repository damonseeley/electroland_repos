#include "Axis.h"
#include <GL/glut.h>

void Axis::render() {
	glPushMatrix();
	glTranslatef(pos.x, pos.y, pos.z);
	glRotatef(rot.x, 1.0,0.0,0.0);
	glRotatef(rot.y, 0.0,1.0,0.0);
	glRotatef(rot.z, 0.0,0.0,1.0);
	glBegin(GL_LINES);
	glColor3f(1.0f,0,0);
	glVertex3f(0,0,0);
	glVertex3f(1.0f,0,0);
	glColor3f(0,1.0,0);
	glVertex3f(0,0,0);
	glVertex3f(0,1.0f,0);
	glColor3f(0,0,1.0f);
	glVertex3f(0,0,0);
	glVertex3f(0,0,1.0f);
	glEnd();
	glPopMatrix();
}