#ifndef __PSYSTEM_H__
#define __PSYSTEM_H__

#include "CinderVector.h"
#include <vector>
#include <windows.h>

using namespace cinder;
using namespace std;
struct Particle {
	Vec3f pos;
	Vec3f vel;
	Vec3f color;
	DWORD endTime;

	Particle *pNext;

};

class PSystem {
public:

//	LUnit texture[i];

	Particle     *m_pActiveList;
    Particle     *m_pFreeList;

	int maxParticles;
	DWORD lifeDuration;
	DWORD lifeVar;
	float flow;  // particles per ms
	Vec3f pos;
	Vec3f initPosVar;
	Vec3f initVel;
	float initVelVar;

//	Vec3f initColor;
	Vec3f initColorVar;
	vector<Vec3f> initColorDist;



	int particleCnt;

	Vec3f gravity;
	Vec3f drag;

	bool addColorsAreRGB;
	bool fadeColorWithDist;
	float minDist;
	float maxDist;



	PSystem();
	~PSystem();


	Vec3f getRandomVector( void );
	float getRandomMinMax( float fMin, float fMax );

	void update(DWORD curTime, float dt);
	void render();
	void addPoints(DWORD curTime, float dt, vector<Vec3f> centers);

};

#endif // __PSYSTEM_H__