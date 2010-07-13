//-----------------------------------------------------------------------------
//           Name: particlesystem.h
//         Author: Kevin Harris
//  Last Modified: 06/03/03
//    Description: Header file for the CParticleSystem Class
//-----------------------------------------------------------------------------

#ifndef CPARTICLESYSTEM_H_INCLUDED
#define CPARTICLESYSTEM_H_INCLUDED

#include "CinderVector.h"

//#include <windows.h>
//#include <gl/gl.h>
//#include <Cg/Cg.h>
//#include <Cg/cgGL.h>
//#include "Vec3f.h"

//using namespace cinder;

//-----------------------------------------------------------------------------
// SYMBOLIC CONSTANTS
//-----------------------------------------------------------------------------

// Classify Point
const int CP_FRONT   = 0;
const int CP_BACK    = 1;
const int CP_ONPLANE = 2;

// Collision Results
const int CR_BOUNCE  = 0;
const int CR_STICK   = 1;
const int CR_RECYCLE = 2;

const float OGL_PI = 3.141592654f;

//-----------------------------------------------------------------------------
// GLOBALS
//-----------------------------------------------------------------------------

struct Plane
{
    Vec3f m_vNormal;           // The plane's normal
    Vec3f m_vPoint;            // A coplanar point within the plane
    float    m_fBounceFactor;     // Coefficient of restitution (or how bouncy the plane is)
    int      m_nCollisionResult;  // What will particles do when they strike the plane

    Plane   *m_pNext;             // Next plane in list
};

struct Particle
{
    Vec3f  m_vCurPos;    // Current position of particle
    Vec3f  m_vCurVel;    // Current velocity of particle
    float     m_fInitTime;  // Time of creation of particle

    Particle *m_pNext;      // Next particle in list
};

struct PointVertex
{
    Vec3f posit;
    Vec3f color;
};

//-----------------------------------------------------------------------------
// CLASSES
//-----------------------------------------------------------------------------

class CParticleSystem
{

public:

    CParticleSystem( void );
   ~CParticleSystem( void );

    enum RENDER_METHOD
    {
       RENDER_METHOD_MANUAL_CPU_BILLBOARDS,
       RENDER_METHOD_ARB_POINT_SPRITES,
       RENDER_METHOD_CG_VERTEX_SHADER
    };

    void SetMaxParticles( DWORD dwMaxParticles ) { m_dwMaxParticles = dwMaxParticles; }
    DWORD GetMaxParticles( void ) { return m_dwMaxParticles; }

    void SetNumToRelease( DWORD dwNumToRelease ) { m_dwNumToRelease = dwNumToRelease; }
    DWORD GetNumToRelease( void ) { return m_dwNumToRelease; }

    void SetReleaseInterval( float fReleaseInterval ) { m_fReleaseInterval = fReleaseInterval; }
    float GetReleaseInterval( void ) { return m_fReleaseInterval; }

    void SetLifeCycle( float fLifeCycle ) { m_fLifeCycle = fLifeCycle; }
    float GetLifeCycle( void ) { return m_fLifeCycle; }

    void SetSize( float fSize ) { m_fSize = fSize; }
    float GetSize( void ) { return m_fSize; }
    float GetMaxPointSize( void ) { return m_fMaxPointSize; }

    void SetColor( Vec3f clrColor ) { m_clrColor = clrColor; }
    Vec3f GetColor( void ) { return m_clrColor; }

    void SetPosition( Vec3f vPosition ) { m_vPosition = vPosition; }
    Vec3f GetPosition( void ) { return m_vPosition; }

    void SetVelocity( Vec3f vVelocity ) { m_vVelocity = vVelocity; }
    Vec3f GetVelocity( void ) { return m_vVelocity; }

    void SetGravity( Vec3f vGravity ) { m_vGravity = vGravity; }
    Vec3f GetGravity( void ) { return m_vGravity; }

    void SetWind( Vec3f vWind ) { m_vWind = vWind; }
    Vec3f GetWind( void ) { return m_vWind; }

    void SetAirResistence( bool bAirResistence ) { m_bAirResistence = bAirResistence; }
    bool GetAirResistence( void ) { return m_bAirResistence; }

    void SetVelocityVar( float fVelocityVar ) { m_fVelocityVar = fVelocityVar; }
    float GetVelocityVar( void ) { return m_fVelocityVar; }

    void SetRenderMethod( RENDER_METHOD renderMethod ) { m_renderMethod =renderMethod; }
    RENDER_METHOD GetRenderMethod( void ) { return m_renderMethod; }

    void SetCollisionPlane( Vec3f vPlaneNormal, Vec3f vPoint, 
                            float fBounceFactor = 1.0f, int nCollisionResult = CR_BOUNCE );

    int Init( void );
    int Update( float fElapsedTime );
    void Render( void );

    void SetTexture( char *chTexFile );
    GLuint GetTextureID( void );

    void RestartParticleSystem( void );

private:

    CGprofile   m_CGprofile;
    CGcontext   m_CGcontext;
    CGprogram   m_CGprogram;
    CGparameter m_CGparam_modelViewProj;
    //CGparameter m_CGparam_size;
    CGparameter m_CGparam_preRotatedQuad;

    Particle     *m_pActiveList;
    Particle     *m_pFreeList;
    Plane        *m_pPlanes;
    DWORD         m_dwActiveCount;
    float         m_fCurrentTime;
    float         m_fLastUpdate;
    float         m_fMaxPointSize;
    bool          m_bDeviceSupportsPSIZE;
    GLuint        m_textureID;
    RENDER_METHOD m_renderMethod;
    
    // Particle Attributes
    DWORD       m_dwMaxParticles;
    DWORD       m_dwNumToRelease;
    float       m_fReleaseInterval;
    float       m_fLifeCycle;
    float       m_fSize;
    Vec3f    m_clrColor;
    Vec3f    m_vPosition;
    Vec3f    m_vVelocity;
    Vec3f    m_vGravity;
    Vec3f    m_vWind;
    bool        m_bAirResistence;
    float       m_fVelocityVar;
    char       *m_chTexFile;
};

#endif /* CPARTICLESYSTEM_H_INCLUDED */