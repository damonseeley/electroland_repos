/*
* Copyright 1993-2010 NVIDIA Corporation.  All rights reserved.
*
* NVIDIA Corporation and its licensors retain all intellectual property and 
* proprietary rights in and to this software and related documentation. 
* Any use, reproduction, disclosure, or distribution of this software 
* and related documentation without an express license agreement from	std::cout << transformMatrix << std::endl;

* NVIDIA Corporation is strictly prohibited.
*
* Please refer to the applicable NVIDIA end user license agreement (EULA) 
* associated with this source code for terms and conditions that govern 
* your use of this NVIDIA software.
* 
*/

/* 
This example demonstrates how to use the Cuda OpenGL bindings to
dynamically modify a vertex buffer using a Cuda kernel.

The steps are:
1. Create an empty vertex buffer object (VBO)
2. Register the VBO with Cuda
3. Map the VBO for writing from Cuda
4. Run Cuda kernel to modify the vertex positions
5. Unmap the VBO
6. Render the results using OpenGL

Host code
*/

#ifdef _WIN32
#  define WINDOWS_LEAN_AND_MEAN
#  define NOMINMAX
#  include <windows.h>
#endif

// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <string.h>
#include <math.h>

// includes, GL
#include <GL/glew.h>

#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif

// includes
#include <cuda.h>
#include <cuda_runtime.h>
#include <cutil_inline.h>
#include <cutil_gl_inline.h>
#include <cutil_gl_error.h>
#include <cuda_gl_interop.h>
#include <vector_types.h>
#include <rendercheck_gl.h>

#include <libconfig.h++>

#include "RenderList.h"
#include "FadeBlock.h"
#include "Guicon.h"
#include "TyzxCam.h"
#include "CloudConstructor.h"
#include "Voxel.h"
#include "PresenceVoxel.h"
#include "PSystem.h"
#include "Axis.h"
#include "Floor.h"
#include "VoxelRenderer.h"

#define PERSP_VIEW 0
#define FRONT_VIEW 1
#define TOP_VIEW 2
#define SIDE_VIEW 3
using namespace std;
using namespace libconfig;

#define MAX_EPSILON_ERROR 10.0f
// Define the files that are to be save and the reference images for validation
const char *sOriginal[] =
{
	"simpleGL.ppm",
	NULL
};

const char *sReference[] =
{
	"ref_simpleGL.ppm",
	NULL
};


////////////////////////////////////////////////////////////////////////////////
// constants
int window_width = 512;
int window_height = 512;
float FOV = 60;
float frustumNear = .1;
float frustumTop = tan(FOV * 3.14159/360.0) * frustumNear;
float frustumAspect = 1.33;
float frustumSide = frustumAspect*frustumTop;

float canvasRatio;
int viewWidth;
int color_depth;
int refresh;
bool isFullscreen = false;
bool showOrtho = false;
bool showAxis = true;
Config config;

RenderList *renderList;

int camCnt;
TyzxCam **tyzxCams;
Vec3f camWorldTrans;
Vec3f camWorldRot;

Vec3f topZero = Vec3f(100,100, 10000);
Vec3f topBoundsMin;
Vec3f topBoundsMax;
Vec3f sideZero;
Vec3f frontZero;


Axis *origin;
Floor *vFloor;

CloudConstructor *cloudConstructor;

Voxel *rawVoxel; // raw
Voxel *bgVoxel;
Voxel *fgVoxel;
Voxel *persistanceVoxel;
Voxel *lastPersistanceVoxel;
Voxel *diffVoxel;


VoxelRenderer *voxelRenderer;


vector<PSystem *> pSystems;

vector<Vec3f> voxCenters;


enum VoxelMode         { RAW_VOXELS,	BG_VOXELS,		FG_VOXELS,		PERSIST_VOXELS,		DIFF_VOXELS,	NO_VOXELS};
char *voxelModeStr[] = {"RAW_VOXELS",	"BG_VOXELS",	"FG_VOXELS",	"PERSIST_VOXELS",	"DIFF_VOXELS",	"NO_VOXELS"};
int voxelMode = RAW_VOXELS;
bool showPoints = true;
float voxelThresh;
float persistanceThresh;
float adaptation;



int selectedCam = -1;	
bool selectedCamExclusive = false;

// vbo variables
//GLuint vbo;
//struct cudaGraphicsResource *cuda_vbo_resource;
//void *d_vbo_buffer = NULL;

float anim = 0.0;

// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0, rotate_y = 0, rotate_z = 0;
//float rotate_x = 90.0, rotate_y = 0.0, rotate_z = -90;
//float translate_z = 1800; // aprox 6 ft
//float translate_x = 3000;
//float translate_y = -900;

float translate_z = 0; // aprox 5 ft
float translate_x = 0;
float translate_y = 0;


//unsigned int timer = 0;

const int frameCheckNumber = 100;
int frameCount = 0;        // FPS count for averaging
float fps = 30.0f;
DWORD lastSystemTime = 0; // use for fps
DWORD lastTime = 0;
DWORD curTime = 0;
float dt = 0;

Vec3d minDim(0,0,0);
Vec3d maxDim(0,0,0);
Vec3d divs(0,0,0);

#define MAX(a,b) ((a > b) ? a : b)

////////////////////////////////////////////////////////////////////////////////
// kernels
//#include <simpleGL_kernel.cu>

//extern "C" 
//void launch_kernel(float4* pos, unsigned int mesh_width, unsigned int mesh_height, float time);
extern "C"
void addTest();
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
//CUTBoolean runTest(int argc, char** argv);
void cleanup();

// GL functionality
CUTBoolean initGL(int argc, char** argv);
CUTBoolean runGL(int argc, char** argv);

//void createVBO(GLuint* vbo, struct cudaGraphicsResource **vbo_res, 
//	       unsigned int vbo_res_flags);
//void deleteVBO(GLuint* vbo, struct cudaGraphicsResource *vbo_res);

// rendering callbacks
void display();
void keyboard(unsigned char key, int x, int y);
void keyboardSpecial(int key, int x, int y);
void mouse(int button, int state, int x, int y);
void motion(int x, int y);

// Cuda functionality
//void runCuda(struct cudaGraphicsResource **vbo_resource);
//void runAutoTest();
//void checkResultCuda(int argc, char** argv, const GLuint& vbo);

const char *sSDKsample = "Guinness Vox";

//char* cam_ip_ar[] = {"192.168.247.48", "192.168.247.75", "192.168.247.41"};



void pauseConsol() {
	std::cout << "Enter . to continue" << std::endl;
	int val;
	while(cin>>val);
}
////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////

int main(int argc, char** argv)

{

	bool configOK = false;
	if(argc > 1) { // if command line arguements
		try {
			config.readFile(argv[1]);
			configOK = true;
		} catch(const FileIOException &fioex) {
			std::cerr << "I/O error while reading file \"" << argv[1] << "\" defaulting to \"GestureTrack.cfg\""<< std::endl;
		} catch(const ParseException &pex) {
			std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
			std::cerr << "defaulting to \"GestureTrack.cfg\"" << std::endl;
		}
	}
	if(! configOK){
		try {
			config.readFile("GestureTrack.cfg");
		} catch(const FileIOException &fioex) {
			std::cerr << "I/O error while reading file \"GestureTrack.cfg\"" << std::endl;
			exit(1);
		} catch(const ParseException &pex) {
			std::cerr << "Parse error at " << pex.getFile() << ":" << pex.getLine() << " - " << pex.getError() << std::endl;
			exit(1);
		}
	}






	runGL(argc, argv);


	//  cutilExit(argc, argv);
}


//const int frameCheckNumber = 100;
//int frameCnt = 0;        // FPS count for averaging
//long lastSystemTime = -1;

void computeFPS(DWORD curTime) {
	if(lastSystemTime == 0) {
		lastSystemTime = curTime;
	} else if (frameCount >= frameCheckNumber) {
		fps = (float)( 1000.0 * (double)frameCount / (double) (curTime - lastSystemTime));
		frameCount = 0;
		lastSystemTime = curTime;
	} else {
		frameCount++;
	}

}
////////////////////////////////////////////////////////////////////////////////
//! Initialize GL
////////////////////////////////////////////////////////////////////////////////

void setupWorld(){
	// INIT CAMS
	// need to get these from init file TODO
	origin = new Axis(Vec3f(0,0,0), Vec3f(0,0,0));

	const Setting& configRoot = config.getRoot();



	if(! configRoot.exists("cameras")) {
		std::cerr << "No camera's specified in config file\n" << std::endl;
		exit(1);
	} 



	const Setting &cameras = configRoot["cameras"];

	camCnt = cameras.getLength();
	tyzxCams = (TyzxCam **) malloc(sizeof(TyzxCam *) * camCnt);

	for(int i =  0; i < camCnt;i++) {
		const Setting& camera = cameras[i];
		string ip;
		string name;
		camera.lookupValue("ipnname", name);
		camera.lookupValue("ip", ip);

		Vec3d trans(0,0,0);
		Vec3d rot(0,0,0);

		//MATRIXXXXX
		if(camera.exists("matrix")) {
			double m[12];
			m[0] = camera["matrix"][0];
			m[1] = camera["matrix"][1];
			m[2] = camera["matrix"][2];
			m[3] = camera["matrix"][3];
			m[4] = camera["matrix"][4]; 
			m[5] = camera["matrix"][5];
			m[6] = camera["matrix"][6];
			m[7] = camera["matrix"][7];
			m[8] = camera["matrix"][8];
			m[9] = camera["matrix"][9];
			m[10] = camera["matrix"][10];
			m[11] = camera["matrix"][11];
			tyzxCams[i] = new TyzxCam(name.c_str(), trans, rot, ip.c_str());
			tyzxCams[i]->setMatrix(
				m[0], m[1], m[2],
				m[3], m[4], m[5],
				m[6], m[7], m[8],
				m[9], m[10], m[11]
			);
	} else {
			trans.x = camera["translation"][0];
			trans.y = camera["translation"][1];
			trans.z = camera["translation"][2];

			rot.x = camera["rotation"][0];
			rot.y = camera["rotation"][1];
			rot.z = camera["rotation"][2];
			tyzxCams[i] = new TyzxCam(name.c_str(), trans, rot, ip.c_str());
		}



		std::cout << "starting "  << name << " ("<< ip << ") " << std::endl;
		if(tyzxCams[i]->start()) {
			std::cout << "  txzxCam started\n"  << std::endl;;
		}
	}

	cloudConstructor = new CloudConstructor(tyzxCams, camCnt);


	camWorldTrans.x = configRoot["worldTransform"]["translation"][0];
	camWorldTrans.y = configRoot["worldTransform"]["translation"][1];
	camWorldTrans.z = configRoot["worldTransform"]["translation"][2];
	camWorldRot.x = configRoot["worldTransform"]["rotation"][0];
	camWorldRot.y = configRoot["worldTransform"]["rotation"][1];
	camWorldRot.z = configRoot["worldTransform"]["rotation"][2];


	float fLevel = configRoot["floor"]["level"];
	float fMinX = configRoot["floor"]["minX"];
	float fMaxX = configRoot["floor"]["maxX"];
	float fDepth = configRoot["floor"]["depth"];
	Vec3f fFrontColor = Vec3f(configRoot["floor"]["frontColor"][0],configRoot["floor"]["frontColor"][1],configRoot["floor"]["frontColor"][2]);
	Vec3f fBackColor = Vec3f(configRoot["floor"]["backColor"][0],configRoot["floor"]["backColor"][1],configRoot["floor"]["backColor"][2]);


	vFloor = new Floor(fLevel, fMinX, fMaxX, fDepth, fBackColor, fFrontColor);


	const Setting &voxSet = configRoot["voxel"];


	minDim.x = voxSet["min"][0];
	minDim.y = voxSet["min"][1];
	minDim.z = voxSet["min"][2];

	maxDim.x = voxSet["max"][0];
	maxDim.y = voxSet["max"][1];
	maxDim.z = voxSet["max"][2];

	divs.x = voxSet["div"][0];
	divs.y = voxSet["div"][1];
	divs.z = voxSet["div"][2];

	

	rawVoxel = new Voxel(minDim, maxDim, divs);
	bgVoxel = new Voxel(minDim, maxDim, divs);
	fgVoxel = new Voxel(minDim, maxDim, divs);
	persistanceVoxel = new Voxel(minDim, maxDim, divs);
	lastPersistanceVoxel = new Voxel(minDim, maxDim, divs);
	diffVoxel = new Voxel(minDim, maxDim, divs);
	voxelRenderer = new VoxelRenderer(rawVoxel);


	voxelThresh = voxSet["thresh"];
	persistanceThresh = voxSet["persistanceThresh"];
	adaptation= voxSet["adaptation"];

	Vec3f boxSize = (maxDim - minDim) /divs;

	const Setting &psystemsConfig = configRoot["psystems"];
	for(int i = 0 ; i < psystemsConfig.getLength(); i++) {
		const Setting& psystemConfig = psystemsConfig[i];

		PSystem *psys = new PSystem();
		psys->maxParticles = psystemConfig["maxParticles"];
		psys->lifeVar= psystemConfig["lifeDuration"];
		psys->lifeDuration = psystemConfig["lifeVar"];
		psys->flow = psystemConfig["flow"];
		psys->initVelVar = psystemConfig["initVelVar"];
		psys->initVel.x = psystemConfig["initVel"][0];
		psys->initVel.y = psystemConfig["initVel"][1];
		psys->initVel.z = psystemConfig["initVel"][2];
		float boxVar = psystemConfig["boxVar"];
		psys->initPosVar = boxSize * 1.1f;

		const Setting &colors = psystemConfig["colors"];
		psys->initColorDist.clear();
		for(int i = 0 ; i < colors.getLength(); i++) {
			Vec3f color = Vec3f(colors[i][0], colors[i][1],colors[i][2]);
			psys->initColorDist.push_back(color);
		}
		psys->initColorVar.x= psystemConfig["initColorVar"][0];
		psys->initColorVar.y= psystemConfig["initColorVar"][1];
		psys->initColorVar.z= psystemConfig["initColorVar"][2];
		psys->gravity.x = psystemConfig["gravity"][0];
		psys->gravity.y = psystemConfig["gravity"][1];
		psys->gravity.z = psystemConfig["gravity"][2];


		psys->minDist = minDim.y + 250;
		psys->maxDist = maxDim.y - 250;

		pSystems.push_back(psys);
	}
	//	pSystem2->maxParticles = 30000;
	///	pSystem2->lifeDuration = 150;
	//	pSystem2->lifeVar = 75;
	//	pSystem2->flow = 10000;  // particles per ms
	//	pSystem1->initVel = Vec3f(0,0,0);
	///	pSystem1->initVelVar = .001;

	//	}

	//	pSysColors1.push_back(Vec3f(200,.99,1));
	//	pSysColors1.push_back(Vec3f(200,.99,1));
	//	pSysColors1.push_back(Vec3f(160,.99,1));

	//	pSysColors2.push_back(Vec3f(160,.99,1));
	//	pSysColors2.push_back(Vec3f(160,.99,1));
	//	pSysColors2.push_back(Vec3f(120,.99,1));

	//	pSystem1->initColorVar= Vec3f(5, .01, 0);
	//	pSystem2->initColorVar= Vec3f(5, .01, 0);


	//	pSystem1->minDist = minDim.y + 250;
	//	pSystem1->maxDist = maxDim.y - 250;
	//	pSystem2->minDist = minDim.y + 250;
	//	pSystem2->maxDist = maxDim.y - 250;

	//	Vec3f boxSize = (maxDim - minDim) /divs;
	//	pSystem1->initPosVar = boxSize * 1.1f;
	//	pSystem2->initPosVar = boxSize * .9f;

	//	pSystem1->maxParticles = 20000;
	//	pSystem1->lifeDuration = 100;
	//	pSystem1->lifeVar = 1000;
	//	pSystem1->flow = 1000;  // particles per ms
	//	pSystem1->initVel = Vec3f(.1,.1,.1);
	//	pSystem1->initVelVar = .025;

	//	pSystem2->maxParticles = 30000;
	///	pSystem2->lifeDuration = 150;
	//	pSystem2->lifeVar = 75;
	//	pSystem2->flow = 10000;  // particles per ms
	//	pSystem1->initVel = Vec3f(0,0,0);
	///	pSystem1->initVelVar = .001;


	//	pSystem->fadeColorWithDist = false;


	renderList = new RenderList();

	//	FadeBlock::createDisplayList((maxDim-minDim)/divs);

	translate_z = -maxDim.z;
	translate_x = (float)  -(maxDim.x + minDim.x) * .5f;
	translate_y = -1524; // approx 5 ft

	curTime = timeGetTime();
	lastSystemTime = curTime;
	lastTime = curTime;


//	topZero(
	topBoundsMin.x =  10000+topZero.x + topZero.z;
			topBoundsMax.x =  -10000+topZero.x - topZero.z;
			topBoundsMin.y =  -10000+topZero.y - topZero.z;
			topBoundsMax.y =  10000+topZero.y+topZero.z;
			topBoundsMin.z =  10000;
			topBoundsMax.z =  -10000;


}


CUTBoolean initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);


	const Setting &window = config.getRoot()["window"];
	window_width = window["width"];
	window_height = window["height"];


	glutInitWindowSize(window_width, window_height);
	glutCreateWindow("Electroland Gesture Track");
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);	
	glutMotionFunc(motion);

	// initialize necessary OpenGL extensions
	glewInit();
	if(window["fullscreen"]) {
		glutGameModeString(window["fullscreenString"]);
		isFullscreen = true;
		glutEnterGameMode();		
	}

	if (! glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
		fflush(stderr);
		return CUTFalse;
	}

	// default initialization
	glClearColor(window["bgColor"][0], window["bgColor"][1], window["bgColor"][2], 1.0);
	glEnable(GL_DEPTH_TEST);
	glEnable (GL_BLEND);
	glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

	canvasRatio = window["canvasRatio"];
	viewWidth = (int) (canvasRatio * window_height);


	FOV = window["FOV"];
	frustumNear = .1;
	frustumTop = tan(FOV * 3.14159/360.0) * frustumNear;
	frustumAspect = canvasRatio;
	frustumSide = frustumAspect*frustumTop;



	// viewport
	//	glViewport(viewWidth, 0, 0, window_height);
	glViewport(0, 0, viewWidth, window_height);

	// projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();


	CUT_CHECK_ERROR_GL();


	//	box = buildBox(Vec3f(1,1,1));

	setupWorld();

	return CUTTrue;
}

CUTBoolean runGL(int argc, char** argv) {
	if (CUTFalse == initGL(argc, argv)) {
		return CUTFalse;
	}

	cudaGLSetGLDevice( cutGetMaxGflopsDeviceId() ); // set the fastest device

	// register callbacks
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);	

	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutMainLoop();

	return CUTTrue;
}
//3224224441
//APN wap.cingular
//username WAP@CINGULARGPRS.COM
//CINGULAR1


////////////////////////////////////////////////////////////////////////////////
//! Display callback
////////////////////////////////////////////////////////////////////////////////

void render(int view)
{
	lastTime = curTime;
	curTime = timeGetTime();
	dt = (float)(curTime - lastTime);
	//cutilCheckError(cutStartTimer(timer));  

	// run CUDA kernel to generate vertex positions
	//    runCuda(&cuda_vbo_resource);

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	// set view matrix
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();


	if(view == PERSP_VIEW) {
		glRotatef(rotate_x, 1.0, 0.0, 0.0);
		glRotatef(rotate_y, 0.0, 1.0, 0.0);
		glRotatef(rotate_z, 0.0, 0.0, 1.0);
		glTranslatef(translate_x, translate_y, translate_z);
	} else if (view == TOP_VIEW) {
		glRotatef(-90, 1.0, 0.0, 0.0);
		//		glRotatef(-90, 0.0, 1.0, 0.0);
		//	glTranslatef(translate_x, translate_y, translate_z);
	} else if (view == FRONT_VIEW) {
		//		glTranslatef(translate_x, translate_y, translate_z);
	} else {// side
		glRotatef(-90, 0.0, 1.0, 0.0);
		//		glTranslatef(translate_x, translate_y, translate_z);
	}






	vFloor->render();


	glColor3f( .50f, .5f, 0.5f );


	glBegin(GL_LINE_STRIP);
	glVertex3f(-65535.0f, -65535.0f, -65535.0f); //1
	glVertex3f(-65535.0f, -65535.0f, 65535.0f); //2
	glVertex3f(65535.0f, -65535.0f, 65535.0f); //3
	glVertex3f(65535.0f, -65535.0f, -65535.0f); //4
	glVertex3f(-65535.0f, -65535.0f, -65535.0f); //1


	glVertex3f(-65535.0f, 65535.0f, -65535.0f); //1
	glVertex3f(-65535.0f, 65535.0f, 65535.0f); //2
	glVertex3f(65535.0f, 65535.0f, 65535.0f); //3
	glVertex3f(65535.0f, 65535.0f, -65535.0f); //4
	glVertex3f(-65535.0f, 65535.0f, -65535.0f); //1


	glVertex3f(-65535.0f, -65535.0f, -65535.0f); //1
	glVertex3f(-65535.0f, -65535.0f, 65535.0f); //2
	glVertex3f(-65535.0f, 65535.0f, 65535.0f); //2
	glVertex3f(65535.0f, 65535.0f, 65535.0f); //3
	glVertex3f(65535.0f, -65535.0f, 65535.0f); //3
	glVertex3f(65535.0f, -65535.0f, -65535.0f); //4
	glVertex3f(65535.0f, 65535.0f, -65535.0f); //4

	glEnd();




	Vec3f zero = Vec3f(0,0,0);
	Vec3f xAxis = Vec3f(100,0,0);
	Vec3f yAxis = Vec3f(0,100,0);
	Vec3f zAxis = Vec3f(0,0,100);

	for(int i =0; i < camCnt; i++) {
		tyzxCams[i]->grab();
		/*
		glBegin(GL_TRIANGLE_FAN);

		glColor3f(1,1,1);
		Vec3f pt = tyzxCams[i]->transformMatrix.transformPoint(zero);
		glVertex3f(pt.x, pt.y, pt.z);

		glColor3f(1,0,0);
		pt = tyzxCams[i]->transformMatrix.transformPoint(xAxis);
		glVertex3f(pt.x, pt.y, pt.z);

		glColor3f(0,1,0);
		pt = tyzxCams[i]->transformMatrix.transformPoint(yAxis);
		glVertex3f(pt.x, pt.y, pt.z);
		
		glColor3f(0,0,1);
		 pt = tyzxCams[i]->transformMatrix.transformPoint(zAxis);
		glVertex3f(pt.x, pt.y, pt.z);

		glColor3f(1,0,0);
		pt = tyzxCams[i]->transformMatrix.transformPoint(xAxis);
		glVertex3f(pt.x, pt.y, pt.z);

		glEnd();
		*/

	}
		glPointSize(1);


	cloudConstructor->calcPoints(false);

	if(showPoints) {

		glPointSize(1.0);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, cloudConstructor->getPoints());
		for(int i = 0; i < camCnt; i++) {
			if ((selectedCam < 0) || (selectedCam == i )) {
				glColor3f(1.0,0.0,0.0);
			} else {
				glColor3f(0.5,0.5,0.5);
			}
			if ((selectedCam <0) || (! selectedCamExclusive) ||(selectedCam == i)) {
				glDrawArrays(GL_POINTS, 3 + (cloudConstructor->imgSize * i),  cloudConstructor->imgSize);
			}
		}
		glDisableClientState(GL_VERTEX_ARRAY);


	}

	rawVoxel->calcVoxel(cloudConstructor->pointCnt, cloudConstructor->getGPUPoints(), true, false);
	cloudConstructor->freeGPUPoints();
	bgVoxel->addScale(adaptation, rawVoxel,1.0f - adaptation, false);
	fgVoxel->sub(rawVoxel, bgVoxel, false);
	rawVoxel->deallocateGridOnGPU();
	bgVoxel->deallocateGridOnGPU();
	persistanceVoxel->incIfOverThresh(fgVoxel, voxelThresh, false);
	fgVoxel->deallocateGridOnGPU();
	diffVoxel->sub(lastPersistanceVoxel, persistanceVoxel,true);
	persistanceVoxel->deallocateGridOnGPU();
	lastPersistanceVoxel->deallocateGridOnGPU();
	lastPersistanceVoxel->copyGrid(persistanceVoxel);	

	// if its in the forground the current is zero then set with current time else set to 

//	enum VoxelMode { RAW_VOXELS, BG_VOXELS, FG_VOXELS, PERSIST_VOXELS, DIFF_VOXELS, NO_VOXELS};

	switch(voxelMode) {
		case RAW_VOXELS:
			rawVoxel->draw(voxelThresh);
			break;
		case BG_VOXELS:
			bgVoxel->draw(voxelThresh);
			break;
		case FG_VOXELS:
			fgVoxel->draw(voxelThresh);
			break;
		case PERSIST_VOXELS:
			std::cout << "render threash " <<  persistanceThresh * fps << std::endl;
			persistanceVoxel->draw(persistanceThresh * fps);			
			break;
		case DIFF_VOXELS:
			diffVoxel->draw(persistanceThresh * fps);			
			break;
		case NO_VOXELS:
			break;
	}

}

/*
	if(voxelMode == NO_VOXELS) {
		curVoxel->deallocateGridOnGPU();
		//		curVoxel->draw(1000.0f);
	} else if(voxelMode == RAW_VOXELS) {
		curVoxel->deallocateGridOnGPU();
		curVoxel->draw(.5f);
	} else {
		bgVoxel->addScale(adaptation, curVoxel,1.0f - adaptation, false);
		if(voxelMode == BG_VOXELS) {
			bgVoxel->deallocateGridOnGPU();
			bgVoxel->draw(.5f);
		} else {
			//			curVoxel->addScale(1, bgVoxel, -1);
			curVoxel->sub(bgVoxel);
			bgVoxel->deallocateGridOnGPU();
			curVoxel->deallocateGridOnGPU();
			if(voxelMode== FG_VOXELS) {
				curVoxel->draw(.75f);
			} else {


				voxCenters.clear();
				Vec3f sides = (curVoxel->maxDim-curVoxel->minDim);
				sides /= curVoxel->divisions;

				float* gridPtr = curVoxel->grid;

				float curZ;
				float curY;

				for(int k = 0; k < curVoxel->divisions.z; k++) {
					curZ = curVoxel->minDim.z + (k+.5) * sides.z;
					for(int j = 0; j < curVoxel->divisions.y; j++) {
						curY = curVoxel->minDim.y+ (j+.5) * sides.y;
						for(int i = 0; i < curVoxel->divisions.x; i++) {
							if(*gridPtr > .25) {
								voxCenters.push_back(Vec3f(curVoxel->minDim.x + (i+.5) * sides.x, curY, curZ));
							}
							gridPtr++;
						}
					}
				}

				for(vector<PSystem *>::iterator it = pSystems.begin(); it!=pSystems.end(); it++) {
					(*it)->addPoints(curTime,dt,voxCenters);
					(*it)->update(curTime,dt);
					(*it)->render();
				}


				/*
				lastVoxel->sub(curVoxel);
				curVoxel->deallocateGridOnGPU();
				if(voxelMode == DIFF_VOXELS) {
				lastVoxel->draw(.5f);
				}

				lastVoxel->copyGrid(curVoxel);

				Vec3f sides = (lastVoxel->maxDim-lastVoxel->minDim);
				sides /= lastVoxel->divisions;

				float* gridPtr = lastVoxel->grid;
				for(int k = 0; k < lastVoxel->divisions.z; k++) {
				for(int j = 0; j < lastVoxel->divisions.y; j++) {
				for(int i = 0; i < lastVoxel->divisions.x; i++) {
				if(*gridPtr > .5) {
				renderList->add(new FadeBlock(lastSystemTime, lastSystemTime+1000, 
				(i+.5) * sides.x, (j+.5) * sides.y, (k+.5) * sides.z));
				}
				gridPtr++;

				}
				}
				}
			}
		}
*/
//		bgVoxel->deallocateGridOnGPU();
//		curVoxel->deallocateGridOnGPU();
//		lastVoxel->deallocateGridOnGPU();


		/*
		if((voxelMode == DIFF_VOXELS) || (voxelMode == PRES_VOXELS)) {
		curVoxel->sub(bgVoxel,false);
		bgVoxel->deallocateGridOnGPU();
		curVoxel->threshSet(voxelThresh,0,1,false);
		presenceVoxel->add(curVoxel, false); // increment cubes
		presenceVoxel->mult(curVoxel, false); // mask		
		lastPresenceVoxel->sub(presenceVoxel); // this should be all the old voxels
		lastPresenceVoxel->copyGrid(presenceVoxel);
		} else {
		}	
		curVoxel->deallocateGridOnGPU();



		if(voxelMode == BG_VOXELS) {
		} else if(voxelMode != NO_VOXELS) {
		presenceVoxel->draw(.5f);
		}
		*/
//	}
	//	renderList->draw(lastSystemTime);

//}

void display() {
	if(showOrtho) {

		glViewport(0,  window_height/2, viewWidth/2, window_height/2);
		glScissor(0,  window_height/2, viewWidth/2, window_height/2);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(frustumSide,-frustumSide, -frustumTop, frustumTop, frustumNear, 1310700);
		render(PERSP_VIEW);


		glViewport (0,0, window_width/2, window_height/2);
		glScissor (0,0, window_width/2, window_height/2);
		glMatrixMode (GL_PROJECTION);								// Select The Projection Matrix
		glLoadIdentity ();
		//	glOrtho ( 1000 - translate_z, - 1000+translate_z,  - 1000+translate_z,  + 1000-translate_z,10000,-10000);
		glOrtho ( 10000+frontZero.x + frontZero.z,     -10000+frontZero.x-frontZero.z, 
			-10000+frontZero.y - frontZero.z,      10000+frontZero.y+frontZero.z, 100000,-100000);
		render(FRONT_VIEW);

		glViewport (window_width/2,window_height/2, window_width/2, window_height/2);
		glScissor (window_width/2,window_height/2, window_width/2, window_height/2);
		glMatrixMode (GL_PROJECTION);								// Select The Projection Matrix
		glLoadIdentity ();
//		glOrtho ( 10000+topZero.x + topZero.z,     -10000+topZero.x-topZero.z, 
//			-10000+topZero.y - topZero.z,      10000+topZero.y+topZero.z, 100000,-100000);
		glOrtho(topBoundsMin.x, topBoundsMax.x,	topBoundsMin.y, topBoundsMax.y,topBoundsMin.z, topBoundsMax.z);
		render(TOP_VIEW);

		glViewport (window_width/2,0, window_width/2, window_height/2);
		glScissor (window_width/2,0, window_width/2, window_height/2);
		glMatrixMode (GL_PROJECTION);								// Select The Projection Matrix
		glLoadIdentity ();
		glOrtho ( 10000+sideZero.x + sideZero.z,     -10000+sideZero.x-sideZero.z, 
			-10000+sideZero.y - sideZero.z,      10000+sideZero.y+sideZero.z, 100000,-100000);
		render(SIDE_VIEW);


	} else {

		glViewport(0, 0, viewWidth, window_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		glFrustum(frustumSide,-frustumSide, -frustumTop, frustumTop, frustumNear, 1310700);
		render(PERSP_VIEW);
	}

	glutSwapBuffers();
	glutPostRedisplay();

	//   cutilCheckError(cutStopTimer(timer));  
	computeFPS(curTime);

}

void cleanup()
{
	// cutilCheckError( cutDeleteTimer( timer));

	//   deleteVBO(&vbo, cuda_vbo_resource);

	//  if (g_CheckRender) {
	//     delete g_CheckRender; g_CheckRender = NULL;
	// }
}

void keyboardSpecial(int key, int /*x*/, int /*y*/)
{
	switch(key) {
	case(GLUT_KEY_UP):
		voxelThresh++;
		std::cout << "Voxel display threshold is now " << voxelThresh << std::endl;
		break;

	case(GLUT_KEY_DOWN):
		voxelThresh--;
		voxelThresh = (voxelThresh < 0) ? 0 : voxelThresh;
		std::cout << "Voxel display threshold is now " << voxelThresh << std::endl;
		break;
	}
}


////////////////////////////////////////////////////////////////////////////////
//! Keyboard events handler
////////////////////////////////////////////////////////////////////////////////
void keyboard(unsigned char key, int /*x*/, int /*y*/)
{
	switch(key) {
	case('?'):
		std::cout << "\n GestrueTrack Help" << std::endl;
		std::cout << "?\t\t- this help " << std::endl;
		std::cout << "esc\t\t- exit " << std::endl;
		std::cout << "z\t\t- display fps" << std::endl;
		std::cout << "p\t\t- hide/show point cloud " << std::endl;
		std::cout << "v\t\t- change voxels display mode" << std::endl;
		std::cout << "=\t\t- show ortho views" << std::endl;
		std::cout << "0-9\t\t- select camera " << std::endl << std::endl;
		std::cout << "-\t\t- deselect camera " << std::endl << std::endl;
		std::cout << "--- With Selected Camera ---" << std::endl;
		std::cout << "translate" << std::endl;
		std::cout << "\t\t w  r" << std::endl;
		std::cout << "\t\tasd f" << std::endl;
		std::cout << "rotate" << std::endl;
		std::cout << "\t\tuio" << std::endl;
		std::cout << "\t\tjkl" << std::endl;
		std::cout << "SPACE\t\t- toggle fullscreen" << std::endl;
		std::cout << "SHIFT and ALT modify speed of rotation and translation" << std::endl;
		std::cout << "Arrow UP/DOWN adjust voxel display threshold" << std::endl;
		std::cout << std::endl;
		break;
	case(27) :
		exit(0);
		break;
	case('z'):
		std::cout << "FPS " << fps << std::endl;
		break;
	case('='):
		if(showOrtho) {
			showOrtho = false;
			glDisable(GL_SCISSOR_TEST);
		} else {
			showOrtho = true;
			glEnable(GL_SCISSOR_TEST);
		}
		break;
	case('-'):
		selectedCamExclusive = false;
		selectedCam = -1;
		break;
	case(' '):
		if(isFullscreen) {
			glutLeaveGameMode();
			isFullscreen = false;
		} else {
			glutEnterGameMode();
			isFullscreen = true;
		}
		break;
	case('p'):
		showPoints = ! showPoints;
		break;
	case('v'):
		voxelMode++;
		if(voxelMode > NO_VOXELS) 
			voxelMode = RAW_VOXELS;
		std::cout << "Displaying raw voxels " << voxelModeStr[voxelMode] << std::endl;
		break;
	default:
		int camNum = key - '1' + 1;
		if(key ==  '0') {
			camNum = 0;
		}
		if((camNum >= 0) && (camNum < camCnt)) {
			if(camNum == selectedCam) {
				selectedCamExclusive = ! selectedCamExclusive;
				if(selectedCamExclusive) {
					std::cout << "Camera " << camNum << " (" << tyzxCams[camNum]->camIP << ")" << " is selected, others cameras HIDDEN" << std::endl;
				} else {
					std::cout << "Camera " << camNum << " (" << tyzxCams[camNum]->camIP << ")" << " is selected, others cameras SHOWING" << std::endl;
				}
			} else {
				selectedCam = camNum;
				selectedCamExclusive = false;
				std::cout << "Camera " << camNum << " (" << tyzxCams[camNum]->camIP << ")" << " is selected, others cameras SHOWING" << std::endl;
			}
		}
	}

	
	if(selectedCam >= 0) {
		switch(key) {
			case('w'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,1);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,100);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('W'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,.1);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,10);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('a'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(1,0,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(100,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('A'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0.1,0,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(10,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('d'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(-1,0,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(-100,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('D'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(-0.1,0,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(-10,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;		
			case('s'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,-1);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,-100);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('S'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,-0.1);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,0,-10);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('r'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,1,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,100,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('R'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,.1,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,10,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('f'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,-1,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,-100,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('F'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->translation += Vec3f(0,-0.1,0);
				} else {
					tyzxCams[selectedCam]->translation += Vec3f(0,-10,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('j'): // yaw
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,.1);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,10);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('J'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,.01);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,1);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;	
			case('l'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,-.1);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,-10);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('L'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,-.01);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0,-1);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('i'): // pitch
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(.1,0,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(10,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('I'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(.01,0,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(10,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;	
			case('k'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(-.1,0,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(-10,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('K'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(-.01,0,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(-1,0,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('o'): // roll
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0.1,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,10,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('O'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(0,0.01,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(0,1,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;	
			case('u'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
						tyzxCams[selectedCam]->rotation += Vec3f(0,-0.1,0);
			} else {
						tyzxCams[selectedCam]->rotation += Vec3f(0,-10,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('U'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
						tyzxCams[selectedCam]->rotation += Vec3f(0,-0.01,0);
				} else {
						tyzxCams[selectedCam]->rotation += Vec3f(0,-1,0);
				}
//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
		}
	tyzxCams[selectedCam]->updateTransform( );

	}/* else { // change world

		switch(key) {
			case('w'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,0,1);
				} else {
					camWorldTrans+=Vec3f(0,0,100);
				}
				break;
			case('W'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,0,.1);
				} else {
					camWorldTrans+=Vec3f(0,0,10);
				}
				break;
			case('a'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(-1,0,0);
				} else {
					camWorldTrans+=Vec3f(-100,0,0);
				}
				break;
			case('A'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(-0.1,0,0);
				} else {
					camWorldTrans+=Vec3f(-10,0,0);
				}
				break;
			case('d'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(1,0,0);
				} else {
					camWorldTrans+=Vec3f(100,0,0);
				}
				break;
			case('D'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(.1,0,0);
				} else {
					camWorldTrans+=Vec3f(10,0,0);
				}
				break;		
			case('s'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,0,-1);
				} else {
					camWorldTrans+=Vec3f(0,0,-100);
				}
				break;
			case('S'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,0,-.1);
				} else {
					camWorldTrans+=Vec3f(0,0,-10);
				}
				break;
			case('r'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,1,0);
				} else {
					camWorldTrans+=Vec3f(0,100,0);
				}
				break;
			case('R'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,0.1,0);
				} else {
					camWorldTrans+=Vec3f(0,10,0);
				}
				break;
			case('f'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,-1,0);
				} else {
					camWorldTrans+=Vec3f(0,-100,0);
				}
				break;
			case('F'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldTrans+=Vec3f(0,-0.1,0);
				} else {
					camWorldTrans+=Vec3f(0,-10,0);
				}
				break;
			case('j'): // yaw
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0,.1);
				} else {
					camWorldRot+=Vec3f(0,0,10);
				}
				break;
			case('J'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0,.01);
				} else {
					camWorldRot+=Vec3f(0,0,1);
				}
				break;	
			case('l'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0,-.1);
				} else {
					camWorldRot+=Vec3f(0,0,-10);
				}
				break;
			case('L'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0,-.01);
				} else {
					camWorldRot+=Vec3f(0,0,-.1);
				}
				break;
			case('i'): // pitch
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0.1,0,0);
				} else {
					camWorldRot+=Vec3f(10,0,0);
				}
				break;
			case('I'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(.01,0,0);
				} else {
					camWorldRot+=Vec3f(1,0,0);
				}
				break;	
			case('k'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(-0.1,0,0);
				} else {
					camWorldRot+=Vec3f(-10,0,0);
				}
				break;
			case('K'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(-0.010,0,0);
				} else {
					camWorldRot+=Vec3f(-1,0,0);
				}
				break;
			case('o'): // roll
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0.1,0);
				} else {
					camWorldRot+=Vec3f(0,10,0);
				}
				break;
			case('O'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,0.01,0);
				} else {
					camWorldRot+=Vec3f(0,1,0);
				}
				break;	
			case('u'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,-0.1,0);
				} else {
					camWorldRot+=Vec3f(0,-10,0);
				}
				break;
			case('U'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					camWorldRot+=Vec3f(0,-0.01,0);
				} else {
					camWorldRot+=Vec3f(0,-1,0);
				}
				break;
		}
		for(int i = 0; i < camCnt; i++) {
			tyzxCams[i]->applyWorldTransforms(camWorldTrans, camWorldRot);	
		}*/
}

//	}
//}

//		tyzxCams[camNum]->isCamOn = ! tyzxCams[camNum]->isCamOn;
//		if(tyzxCams[camNum]->isCamOn) {
//			std::cout << "Camera " << camNum << " (" << tyzxCams[camNum]->camIP << ")" << " is ON\n";
//		} else {
//			std::cout << "Camera " << camNum << " (" << tyzxCams[camNum]->camIP << ")" << " is OFF\n";
//		}
//	}
//}

////////////////////////////////////////////////////////////////////////////////
//! Mouse event handlers
////////////////////////////////////////////////////////////////////////////////

float downX;
float downY;
bool wasUp = true;

void mouse(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1<<button;
		if(wasUp) {
			downX = x;
			downY = y;
			wasUp = false;
		}

	} else if (state == GLUT_UP) {
		mouse_buttons = 0;
		if(! wasUp) {
			wasUp = true;
			if((x == downX) && (y == downY)) {
				bool top = y < window_height/2;
				bool right = x> window_width/2;
				if(top && right) {

					Vec3f worldViewportDims = topBoundsMax - topBoundsMin;

					float viewportWidth = window_width/2;
					float worldX =topBoundsMin.x +  ((x-viewportWidth) / viewportWidth) * worldViewportDims.x;

					float viewportHeight = window_height/2;
					float worldZ =  -((y / viewportHeight) * worldViewportDims.y)-topBoundsMax.y;
					std::cout << y << "  world ( " << worldX << ", " << worldZ << " )" << endl;


				}
			}
		}
	}

	mouse_old_x = x;
	mouse_old_y = y;
	glutPostRedisplay();
}

void motion(int x, int y)
{
	float dx, dy;
	dx = x - mouse_old_x;
	dy = y - mouse_old_y;

	if(showOrtho) {
		bool top = y < window_height/2;
		bool left = x< window_width/2;
		if(top && left) {
			if (mouse_buttons & 1) { //left
				translate_z += dy * 100;
				translate_x += dx * 100;
			} else if (mouse_buttons & 4) { //center
				translate_y += dy * 100;
			}  else if (mouse_buttons & 2)   { //right
				rotate_x += dy * 0.2;
				rotate_z += dx * 0.2;
			}
		} else if(top) { // top right
			if (mouse_buttons & 1) { 
				topZero.x += dx * 100;
				topZero.y += dy * 100;
			}  else {
				topZero.z += dy * 100;
			}
			topBoundsMin.x =  10000+topZero.x + topZero.z;
			topBoundsMax.x =  -10000+topZero.x - topZero.z;
			topBoundsMin.y =  -10000+topZero.y - topZero.z;
			topBoundsMax.y =  10000+topZero.y+topZero.z;
			topBoundsMin.z =  10000;
			topBoundsMax.z =  -10000;

			std::cout << "min " << topBoundsMin << std::endl;
			std::cout << "max " << topBoundsMax << std::endl;
			std::cout<<  "diff " << topBoundsMax-topBoundsMin << std::endl;

		} else if(left) { // bottom left
			if (mouse_buttons & 1) { 
				frontZero.x += dx * 100;
				frontZero.y += dy * 100;
			}  else {
				frontZero.z += dy * 100;
			}
		} else {
			if (mouse_buttons & 1) { 
				sideZero.x += dx * 100;
				sideZero.y += dy * 100;
			}  else {
				sideZero.z += dy * 100;
			}
		}

	} else {
		if (mouse_buttons & 1) { //left
			translate_z += dy * 100;
			translate_x += dx * 100;
		} else if (mouse_buttons & 4) { //center
			translate_y += dy * 100;
		}  else if (mouse_buttons & 2)   { //right
			rotate_x += dy * 0.2;
			rotate_z += dx * 0.2;
		}
	}

//	cout << "Mouse: T(" << translate_x << ", " << translate_y << ", " << translate_z << " )  R(" << rotate_x << ", " << rotate_y << ")\n";

	mouse_old_x = x;
	mouse_old_y = y;
}

