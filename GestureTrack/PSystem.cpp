#include "PSystem.h"
#include "GestureTypeDefs.h"
#include <GL/glew.h>
#include <GL/glut.h>
#include "Util.h"

PSystem::PSystem() {
	maxParticles = 25000;
	lifeDuration = 100;
	lifeVar = 1000;
	flow = 1000;  // particles per ms
	pos = Vec3f(0,0,0);
	initPosVar = Vec3f(5,5,5);
	initVel = Vec3f(0,0,0);
	initColorDist.push_back(Vec3f(1.0f, 1.0f, 1.0f));
//	initColor =Vec3f(.5,0,0);
	initColorVar = Vec3f(.25, .25,.25);
	particleCnt = 0;
	initVelVar = .05f;
	m_pActiveList = NULL;
	m_pFreeList = NULL;
	gravity = Vec3f(0,0, 0);
	drag = Vec3f(.9f, .9f, .9f);

	addColorsAreRGB = false;
	fadeColorWithDist = true;
	minDist = 0;
	maxDist = 6000;

}
PSystem::~PSystem() {
	 while( m_pActiveList )
    {
        Particle* pParticle = m_pActiveList;
        m_pActiveList = pParticle->pNext;
        delete pParticle;
    }
    m_pActiveList = NULL;

    while( m_pFreeList )
    {
        Particle *pParticle = m_pFreeList;
        m_pFreeList = pParticle->pNext;
        delete pParticle;
    }
    m_pFreeList = NULL;
}


void PSystem::update(DWORD curTime, float dt){

//	addPoints(curTime, dt, cents);


	Particle  *curParticle, **nextParticle;
	
	nextParticle = &m_pActiveList;
	while( *nextParticle ) {
		curParticle = *nextParticle;
		if(curParticle->endTime <= curTime) {
			*nextParticle = curParticle->pNext;
            curParticle->pNext = m_pFreeList;
            m_pFreeList = curParticle;
            --particleCnt;
		} else {
			
			// apply forces here to velocity here

			curParticle->vel += gravity;
			curParticle->vel *= drag;

			curParticle->pos += curParticle->vel * dt;

			nextParticle = &curParticle->pNext;
		}
	}


}

float PSystem::getRandomMinMax( float fMin, float fMax )
{
    float fRandNum = (float)rand () / RAND_MAX;
    return fMin + (fMax - fMin) * fRandNum;
}

//-----------------------------------------------------------------------------
// Name: getRandomVector()
// Desc: Generates a random vector where X,Y, and Z components are between
//       -1.0 and 1.0
//-----------------------------------------------------------------------------
Vec3f PSystem::getRandomVector( void )
{
    Vec3f vVector;

    // Pick a random Z between -1.0f and 1.0f.
    vVector.z = getRandomMinMax( -1.0f, 1.0f );
    
    // Get radius of this circle
    float radius = (float)sqrt(1 - vVector.z * vVector.z);
    
    // Pick a random point on a circle.
    float t = getRandomMinMax( -PI, PI );

    // Compute matching X and Y for our Z.
    vVector.x = (float)cosf(t) * radius;
    vVector.y = (float)sinf(t) * radius;

    return vVector;
}


void PSystem::render() {
	Particle  *pParticle = m_pActiveList;
	glBegin(GL_POINTS);
	while( pParticle ) {
		glColor3f(pParticle->color.x, pParticle->color.y, pParticle->color.z);
		glVertex3f(pParticle->pos.x, pParticle->pos.y, pParticle->pos.z);
		pParticle=pParticle->pNext;
	}
	glEnd();
}


void PSystem::addPoints(DWORD curTime, float dt, vector<Vec3f> centers) {
	if(centers.size() == 0) return;
		Particle  *newParticle;
		if(particleCnt < maxParticles) {
		int avaiableToEmit = maxParticles - particleCnt;
		int toEmit = (int) (dt * flow);
		toEmit = (toEmit < avaiableToEmit) ? toEmit : avaiableToEmit;
		int endTime = curTime + lifeDuration;
		for(int i = 0; i < toEmit; i++) {
			if(m_pFreeList) {
				newParticle = m_pFreeList;
				m_pFreeList = newParticle->pNext;
			} else {
				newParticle = new Particle;
			}
			newParticle->pNext = m_pActiveList;
			m_pActiveList = newParticle;
			newParticle->vel = initVel;
			if( initVelVar != 0.0f ) {
                Vec3f randomVec = getRandomVector();
                 newParticle->vel += randomVec * initVelVar;
            }

			newParticle->endTime  = curTime + lifeDuration + (DWORD) (rand() % lifeVar) ;
			newParticle->pos    =  centers[rand() % centers.size()] + (getRandomVector() *initPosVar);

//			addColorsAreRGB = false;
//	fadeColorWithDist = true;
//	minDist = 0;
//	maxDist = 6000;


			newParticle->color = initColorDist[rand() % initColorDist.size()];
			Vec3f randomVec = getRandomVector();
			newParticle->color += randomVec * initColorVar;
			if(! addColorsAreRGB) {
				if(fadeColorWithDist) {
					newParticle->color.z *= (maxDist - newParticle->pos.y) / (maxDist-minDist);
					newParticle->color.z = (newParticle->color.z > 1.0) ? 1.0 : newParticle->color.z;
					newParticle->color.z = (newParticle->color.z < 0.0) ? 0.0 : newParticle->color.z;

				}
				newParticle->color = Util::HSV2RGB(newParticle->color );
			}
                
                ++particleCnt;

		}

	}

}
