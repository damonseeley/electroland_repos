#ifndef __FADE_BLOCK_H__
#define __FADE_BLOCK_H__


#include "CinderVector.h"
#include <GL/glut.h>
#include <windows.h>
#include "Drawable.h"

using namespace cinder;

class FadeBlock : public Drawable {
public:

	static GLuint displayList;

	float x,y,z;
	float r,g,b;
	DWORD startTime;
	DWORD endTime;
	float timeScale;
	FadeBlock(DWORD startTime, DWORD endTime, float x, float y, float z, float r,float g, float b);
	static void createDisplayList(Vec3f sides);
	virtual bool draw(DWORD curTime);
	~FadeBlock() {};

}
;
#endif