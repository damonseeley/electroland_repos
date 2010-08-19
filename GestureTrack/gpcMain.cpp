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

#  include <windows.h>

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
#include <sstream>
#include <iostream>

#include <libconfig.h++>

#include "Globals.h"
#include "Guicon.h"
#include "TyzxCam.h"
#include "PersonTrackReceiver.h"
#include "CloudConstructor.h"
#include "Floor.h"
#include "Voxel.h"
#include "UDPSender.h"
#include "TrackHash.h"
#include "ThreadedTrackGrab.h"
#include "FadeBlock.h"

#include "Projection.h"
#include "PersonDetector.h"

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

ThreadedTrackGrab *trackGrab;
TrackHash *trackHash = NULL;
//UDPSender *udpSender = NULL;

int camCnt;
TyzxCam **tyzxCams;
Vec3f camWorldTrans;
Vec3f camWorldRot;

Vec3f topZero = Vec3f(1.5,1.5, 10);
Vec3f topBoundsMin;
Vec3f topBoundsMax;
Vec3f sideZero;
Vec3f frontZero;


CloudConstructor *cloudConstructor;

enum VoxelMode         { RAW_VOXELS,	BG_VOXELS,		FG_VOXELS,		FG_DESPEC_VOXELS,		DIFF_VOXELS,	HALF_VOXELS,   QUATER_VOXELS, RUNNING, NO_VOXELS};
char *voxelModeStr[] = {"RAW_VOXELS",	"BG_VOXELS",	"FG_VOXELS",	"FG_DESPEC_VOXELS",	"DIFF_VOXELS",	"HALF_VOXELS", "QUATER_VOXELS", "RUNNING", "NO_VOXELS"};

enum DetectMode			{ HEIGHT_MAP,	HEAD_CONV,		LOCAL_MAX,		HANDS_AND_HEADS,	NO_DETECT};
char *detectModeStr[] =	{"HEIGHT_MAP",	"HEAD_CONV",	"LOCAL_MAX",	"HANDS_AND_HEADS",	"NO_DETECT"};

int voxelMode = RUNNING;
int detectMode = NO_DETECT;

float voxelThresh = 5;
float voxelDespecThresh = 5;
float voxelThreshSM = 5;
float voxelDespecThreshSM = 5;
float adaptation;
float curAdaptation;

Voxel *rawVoxel;
Voxel *bgVoxel;
Voxel *fgVoxel;
Voxel *fgDespecVoxel;

Voxel *rawVoxelSM;
Voxel *bgVoxelSM;
Voxel *fgVoxelSM;
Voxel *fgDespecVoxelSM;

Projection *projection;
PersonDetector *personDetector;
Floor *vFloor;

//PersonDetector *personDetector;

float minPointSize;
float maxPointSize;
float pointSizeRate;
float curPointSize;


bool worldCamRot = false;




int selectedCam = -1;	
bool selectedCamExclusive = false;


// mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0, rotate_y = 0, rotate_z = 0;

float translate_z = 0;
float translate_x = 0;
float translate_y = 0;

bool cullOn = true;
float cylCutX;
float cylCutZ;
float cylCutR;
float cylCutCeilingHack;

const int frameCheckNumber = 100;
long curFrame = 0;
int frameCount = 0;        // FPS count for averaging
float fps = 30.0f;
DWORD lastSystemTime = 0; // use for fps
DWORD lastTime = 0;
DWORD curTime = 0;
float dt = 0;

ostringstream netMsg;


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
void reshape(int x, int y);


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

//80 (with target at 45 deg)
//After world adjust: angles 37.934022 -6.363750 -8.569449  translation 247.395505 1318.255910 -1354.591601


//After world adjust: angles 21.276537 3.169888 6.171873  translation 28.121747 1381.796529 -2027.356740

// vsm 2
//x 192.2
// y 104.7
// z 1.132
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

	netMsg << fixed;
	netMsg.precision(0);

	const Setting	& configRoot = config.getRoot();

	if(configRoot.exists("net")) {
		PersonTrackReceiver *tracker  = NULL;
		UDPSender *udpSender = NULL;
		const Setting &net = configRoot["net"];
		if(net["useTrackAPI"]) {
			string modIP;
			net.lookupValue("moderatorIP", modIP);
			tracker = new PersonTrackReceiver((const char*) modIP.c_str());
			tracker->setScale((double) net["trackScale"]);
			std::cout << "Starting Person Track" << std::endl;
		}

		if(net["sendTracks"]) {
			string sendIP;
			net.lookupValue("sendIP", sendIP);
			udpSender = new UDPSender(sendIP, net["sendPort"]);
		}

		trackGrab = new ThreadedTrackGrab(tracker, udpSender);

		trackGrab->start();

	}

	if(configRoot.exists("cylinderCull")) {
		const Setting &cCull = configRoot["cylinderCull"];
		if(cCull["cullOn"]) {
			cullOn = true;
			cylCutX = cCull["x"];
			cylCutZ = cCull["z"];
			cylCutR = cCull["r"];
			cylCutCeilingHack = cCull["ceilingCutHack"];
		} else {
			cullOn = false;
		}

	}


	//useTrackAPI = true;
	//	clientIP = 192.168.247.1;
	//	listenPort = 2345;
	//	sendPort = 1234;
	//	int sendIP




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
		double scale = 1;

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
			tyzxCams[i] = new TyzxCam(name.c_str(), trans, rot, scale, ip.c_str());

			bool transpose = false;
			if(camera.exists("transpose")) {
				transpose = camera["transpose"];
			}
			if(transpose) {
				tyzxCams[i]->setMatrix(
					m[0], m[3], m[6],
					m[1], m[4], m[7],
					m[2], m[5], m[8],
					m[9], m[10], m[11]
				);
			} else {
				tyzxCams[i]->setMatrix(
					m[0], m[1], m[2],
					m[3], m[4], m[5],
					m[6], m[7], m[8],
					m[9], m[10], m[11]
				);
			}
		} else {
			trans.x = camera["translation"][0];
			trans.y = camera["translation"][1];
			trans.z = camera["translation"][2];

			rot.x = camera["rotation"][0];
			rot.y = camera["rotation"][1];
			rot.z = camera["rotation"][2];
			tyzxCams[i] = new TyzxCam(name.c_str(), trans, rot, camera["scale"], ip.c_str());
		}



		std::cout << "starting "  << name << " ("<< ip << ") " << std::endl;
		if(tyzxCams[i]->start()) {
			std::cout << "  txzxCam started\n"  << std::endl;;
		}
	}



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
	vFloor = new Floor(fLevel, fMinX, fMaxX, fDepth, fBackColor, fFrontColor, 10, 10);

	cloudConstructor = new CloudConstructor(tyzxCams, camCnt);

	const Setting &voxSet = configRoot["voxel"];
	Vec3f minDim;
	Vec3f maxDim;
	Vec3f divs;
	minDim.x = voxSet["min"][0];
	minDim.y = voxSet["min"][1];
	minDim.z = voxSet["min"][2];

	maxDim.x = voxSet["max"][0];
	maxDim.y = voxSet["max"][1];
	maxDim.z = voxSet["max"][2];

	divs.x = voxSet["div"][0];
	divs.y = voxSet["div"][1];
	divs.z = voxSet["div"][2];

	Vec3f divsSM;

	divsSM.x = voxSet["divCoarse"][0];
	divsSM.y = voxSet["divCoarse"][1];
	divsSM.z = voxSet["divCoarse"][2];


	rawVoxelSM = new Voxel(minDim, maxDim, divsSM);
	bgVoxelSM = new Voxel(minDim, maxDim, divsSM);
	fgVoxelSM = new Voxel(minDim, maxDim, divsSM);
	fgDespecVoxelSM= new Voxel(minDim, maxDim, divsSM);


	rawVoxel = new Voxel(minDim, maxDim, divs);
	bgVoxel = new Voxel(minDim, maxDim, divs);
	fgVoxel = new Voxel(minDim, maxDim, divs);
	fgDespecVoxel= new Voxel(minDim, maxDim, divs);

	FadeBlock::createDisplayList((maxDim-minDim)/divs);



	voxelThresh = voxSet["thresh"];
	voxelDespecThresh = voxSet["despecThresh"];

	voxelThreshSM = voxSet["threshCoarse"];
	voxelDespecThreshSM = voxSet["despecThreshCoarse"];
	adaptation = voxSet["bgAdaptation"];
	curAdaptation = .5f;


//	projection = new Projection(divsSM);
	projection = new Projection(divs);
	std::cout << "minDim " << minDim << "    maxDim " << maxDim << std::endl;

	projection->setRenderLocation(minDim, maxDim); // draw on ceiling

	personDetector = new PersonDetector(projection);
	





	




	translate_x = -(float) configRoot["window"]["translation"][0] ;
	translate_y = - (float)  configRoot["window"]["translation"][1]; // approx 5 ft
	translate_z = - (float) configRoot["window"]["translation"][2] ; // why wasn't this 0.0?

	rotate_x = (float) configRoot["window"]["rotation"][0];
	rotate_y = - (float) configRoot["window"]["rotation"][1]; 
	rotate_z = (float) configRoot["window"]["rotation"][2]; 


	curTime = timeGetTime();
	lastSystemTime = curTime;
	lastTime = curTime;


	//	topZero(
	topBoundsMin.x =  10  +topZero.x + topZero.z;
	topBoundsMax.x =  -10 +topZero.x - topZero.z;
	topBoundsMin.y =  -10 +topZero.y - topZero.z;
	topBoundsMax.y =  10  +topZero.y +topZero.z;
	topBoundsMin.z =  10;
	topBoundsMax.z =  -10;


}


CUTBoolean initGL(int argc, char **argv)
{
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
	const Setting &window = config.getRoot()["window"];

	window_width = window["width"];
	window_height = window["height"];

	if(window["fullscreen"]) {
		glutGameModeString(window["fullscreenString"]);
		isFullscreen = true;
		glutEnterGameMode();		
	} else {
		glutInitWindowSize(window_width, window_height);
		glutCreateWindow("Electroland Gesture Track");
	}
	if(window["hideCursor"]) {
		glutSetCursor(GLUT_CURSOR_NONE);
	}



	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);	
	glutMotionFunc(motion);
	glutReshapeFunc(reshape);
	// initialize necessary OpenGL extensions
	glewInit();




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
	glEnable(GL_POINT_SMOOTH);
	glEnable(GL_LINE_SMOOTH);
	//glEnable(GL_CULL_FACE);	

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
	glutDisplayFunc(display);	std::cout << "loop called" << std::endl;

	glutKeyboardFunc(keyboard);
	glutSpecialFunc(keyboardSpecial);	

	glutMouseFunc(mouse);
	glutMotionFunc(motion);

	glutMainLoop();

	return CUTTrue;
}


void processVoxels(Voxel *raw, Voxel *bg, Voxel *fg, Voxel *despec, float adapt, float thresh, float despectThresh) {
	bg->addScale(1.0-adapt, raw, adapt, false);
	fg->setMask(raw, bg, thresh, false);
	raw->deallocateGridOnGPU();
	bg->deallocateGridOnGPU();
	despec->setNoiseFilter(fg, despectThresh, false);
	fg->deallocateGridOnGPU();
}

void calculate() {
	if(camCnt > 0) {
	cloudConstructor->calcPoints(false);

	if(cullOn) 
		cloudConstructor->cullCylinder(cylCutX,cylCutZ, cylCutR, cylCutCeilingHack);

	rawVoxelSM->calcVoxel(cloudConstructor->pointCnt, cloudConstructor->getGPUPoints(), true, true);
	rawVoxel->calcVoxel(cloudConstructor->pointCnt, cloudConstructor->getGPUPoints(), true, false);
	cloudConstructor->freeGPUPoints();

	curAdaptation -= .001;
	curAdaptation = (curAdaptation < adaptation) ? adaptation : curAdaptation;
	processVoxels(rawVoxel, bgVoxel, fgVoxel, fgDespecVoxel, curAdaptation, voxelThresh, voxelDespecThresh);
	projection->calcProjection(fgDespecVoxel, voxelThresh, false);

	fgDespecVoxel->deallocateGridOnGPU();

	processVoxels(rawVoxelSM, bgVoxelSM, fgVoxelSM, fgDespecVoxelSM, adaptation, voxelThreshSM, voxelDespecThreshSM);

//	projection->calcProjection(fgDespecVoxelSM, voxelThreshSM, false);
	fgDespecVoxelSM->deallocateGridOnGPU();

	projection->deallocateGridOnGPU();

	personDetector->calc(curFrame++);
	}

	if(trackGrab) {
		trackHash = trackGrab->getCurrentHash();
	}

	if(camCnt > 0) {
	trackHash->merge(&personDetector->existingTracks, 1, curFrame);
	}

	if(trackGrab) {
		trackGrab->trackHashUpdated();
	}

}

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
		//std::cout << translate_x << endl;
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


	if(cullOn) {
		if(selectedCam >= 0) {
			glColor3f(1.0,0.0,1.0f);
			glLineWidth(5.0f);
			glBegin(GL_LINES) ;
			glVertex3f(cylCutX-cylCutR, 0, cylCutZ);
			glVertex3f(cylCutX+cylCutR, 0, cylCutZ);
			glVertex3f(cylCutX, 0, cylCutZ-cylCutR);
			glVertex3f(cylCutX, 0, cylCutZ+cylCutR);
			glEnd();
			glLineWidth(1.0f);
		}
	}



	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, cloudConstructor->getPoints());

	if(selectedCam >= 0) {
		for(int i = 0; i < camCnt; i++) {
			if(showAxis) {
				Vec3f zero = Vec3f(0,0,0);
				Vec3f xAxis = Vec3f(100,0,0);
				Vec3f yAxis = Vec3f(0,100,0);
				Vec3f zAxis = Vec3f(0,0,100);
				tyzxCams[i]->transfromPoint(zero);
				tyzxCams[i]->transfromPoint(xAxis);
				tyzxCams[i]->transfromPoint(yAxis);
				tyzxCams[i]->transfromPoint(zAxis);

				glBegin(GL_TRIANGLE_FAN);

				glColor3f(1,1,1);
				glVertex3f(zero.x, zero.y, zero.z);

				glColor3f(1,0,0);
				glVertex3f(xAxis.x, xAxis.y, xAxis.z);

				glColor3f(0,1,0);
				glVertex3f(yAxis.x, yAxis.y, yAxis.z);

				glColor3f(0,0,1);
				glVertex3f(zAxis.x, zAxis.y, zAxis.z);

				glColor3f(1,0,0);
				glVertex3f(xAxis.x, xAxis.y, xAxis.z);

				glEnd();
			}

			if ((selectedCam < 0) || (selectedCam == i )) {
				glColor3f(1.0,0.0,0.0);
			} else {
				glColor3f(0.5,0.5,0.5);
			}
			if ((selectedCam <0) || (! selectedCamExclusive) ||(selectedCam == i)) {
				glPointSize(1.0f);
				glDrawArrays(GL_POINTS, 3 + (cloudConstructor->imgSize * i),  cloudConstructor->imgSize);
			}
		}
	} else {
				glColor3f(.9f,0.55f,0.1f);
		
			glDrawArrays(GL_POINTS, 3,  cloudConstructor->imgSize * camCnt);
		}
			
		

	glDisableClientState(GL_VERTEX_ARRAY);


	switch(voxelMode) {
	case RAW_VOXELS:
		rawVoxel->draw(voxelThresh);
		break;
	case BG_VOXELS:
		bgVoxel->draw(voxelThresh);
		break;
	case FG_VOXELS:
		fgVoxel->draw(0);
		break;
	case FG_DESPEC_VOXELS:
		fgDespecVoxel->draw(0);
	default:
		break;
	}

	switch(detectMode) {
		case HEIGHT_MAP:
			personDetector->render(personDetector->heights);
			break;
		case HEAD_CONV:
			personDetector->render(personDetector->conv);
			break;
		case LOCAL_MAX:
			personDetector->render(personDetector->localMax);
			break;
		case HANDS_AND_HEADS:
			personDetector->render(NULL);
		case NO_DETECT:
		default:
			break;
	}

	trackHash->render();

}


void reshape(int w, int h) {
	std::cout << "reshape"<<std::endl;
	window_height = h;
	window_width = w;
	viewWidth = (int) (canvasRatio * window_height);
}

void display() {
	calculate();
	if(showOrtho) {

		glViewport(0,  window_height/2, viewWidth/2, window_height/2);
		glScissor(0,  window_height/2, viewWidth/2, window_height/2);
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(FOV,(int) ((float) viewWidth / (float) window_height),0.01f,10000.0f);
		//		glFrustum(frustumSide,-frustumSide, -frustumTop, frustumTop, frustumNear, 1310700);
		render(PERSP_VIEW);


		glViewport (0,0, window_width/2, window_height/2);
		glScissor (0,0, window_width/2, window_height/2);
		glMatrixMode (GL_PROJECTION);								// Select The Projection Matrix
		glLoadIdentity ();
		//	glOrtho ( 1000 - translate_z, - 1000+translate_z,  - 1000+translate_z,  + 1000-translate_z,10000,-10000);
		glOrtho ( 10+frontZero.x + frontZero.z,     -10+frontZero.x-frontZero.z, 
			-10+frontZero.y - frontZero.z,      10+frontZero.y+frontZero.z, 10,-10);
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
		glOrtho ( 10+sideZero.x + sideZero.z,     -10+sideZero.x-sideZero.z, 
			-10+sideZero.y - sideZero.z,      10+sideZero.y+sideZero.z, 10,-10);
		render(SIDE_VIEW);


	} else {

		glViewport(0, 0, viewWidth, window_height);

		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		gluPerspective(FOV,(int) ((float) viewWidth / (float) window_height),0.01f,10000.0f);


		//		glFrustum(frustumSide,-frustumSide, -frustumTop, frustumTop, frustumNear, 1310700);
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
		voxelThresh+=.25f;
		std::cout << "Voxel display threshold is now " << voxelThresh << std::endl;
		break;

	case(GLUT_KEY_DOWN):
		voxelThresh-=.25f;
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
		std::cout << "p\t\t- toggle between points and squares " << std::endl;
		std::cout << "v\t\t- turn on/of culling " << std::endl;
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
		std::cout << "Arrow UP/DOWN adjust point/square size" << std::endl;
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

		break;
	case('b'):
		detectMode++;
		if(detectMode > NO_DETECT) 
			detectMode = HEIGHT_MAP;
		std::cout << "Displaying projection " << detectModeStr[detectMode] << std::endl;
		break;
	case('v'):
		voxelMode++;
		if(voxelMode > NO_VOXELS) 
			voxelMode = RAW_VOXELS;
		std::cout << "Displaying raw voxels " << voxelModeStr[voxelMode] << std::endl;
		break;
	case('['):
		worldCamRot = ! worldCamRot;
		if(worldCamRot) {
			std::cout << "rotating cameras in world coords" << std::endl;
		} else {
			std::cout << "rotating cameras in cam coords" << std::endl;
		}
		break;
	case('c'):
		cullOn = ! cullOn;
		std::cout << "cull is " << cullOn << std::endl;
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
				if(worldCamRot) {
					tyzxCams[selectedCam]->applyRotation(Vec3f(0, 10.0, 0.0));
				} else {

					if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
						tyzxCams[selectedCam]->rotation += Vec3f(0,0,.1);
					} else {
						tyzxCams[selectedCam]->rotation += Vec3f(0,0,10);
					}
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
				if(worldCamRot) {
					tyzxCams[selectedCam]->applyRotation(Vec3f(10, 0, 0.0));
				} else {
					if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
						tyzxCams[selectedCam]->rotation += Vec3f(.1,0,0);
					} else {
						tyzxCams[selectedCam]->rotation += Vec3f(10,0,0);
					}
				}
				//				tyzxCams[selectedCam]->applyWorldTransforms(camWorldTrans, camWorldRot);	
				break;
			case('I'):
				if(glutGetModifiers()==GLUT_ACTIVE_ALT) {
					tyzxCams[selectedCam]->rotation += Vec3f(.01,0,0);
				} else {
					tyzxCams[selectedCam]->rotation += Vec3f(1,0,0);
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

	}
}
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
				translate_z += dy *.1f;
				translate_x += dx *.1f;
			} else if (mouse_buttons & 4) { //center
				translate_y += dy * 100;
			}  else if (mouse_buttons & 2)   { //right
				rotate_x += dy * 0.2;
				rotate_z += dx * 0.2;
			}
		} else if(top) { // top right
			if (mouse_buttons & 1) { 
				topZero.x += dx *.1f;
				topZero.y += dy *.1f;
			}  else {
				topZero.z += dy *.1f;
			}
			topBoundsMin.x =  10+topZero.x + topZero.z;
			topBoundsMax.x =  -10+topZero.x - topZero.z;
			topBoundsMin.y =  -10+topZero.y - topZero.z;
			topBoundsMax.y =  10+topZero.y+topZero.z;
			topBoundsMin.z =  10;
			topBoundsMax.z =  -10;

			std::cout << "min " << topBoundsMin << std::endl;
			std::cout << "max " << topBoundsMax << std::endl;
			std::cout<<  "diff " << topBoundsMax-topBoundsMin << std::endl;

		} else if(left) { // bottom left
			if (mouse_buttons & 1) { 
				frontZero.x += dx *.1f;
				frontZero.y += dy *.1f;
			}  else {
				frontZero.z += dy *.1f;
			}
		} else {
			if (mouse_buttons & 1) { 
				sideZero.x += dx*.1f ;
				sideZero.y += dy*.1f ;
			}  else {
				sideZero.z += dy *.1f;
			}
		}

	} else {
		if (mouse_buttons & 1) { //left
			translate_z += dy *.1f;
			translate_x += dx*.1f ;
		} else if (mouse_buttons & 4) { //center
			translate_y += dy *.1f;
		}  else if (mouse_buttons & 2)   { //right
			rotate_x += dy * 0.2;
			rotate_z += dx * 0.2;
		}
		std::cout << "VCAM T" << translate_x << ", " << translate_y << ", " << translate_z << std::endl;
		std::cout << "VCAM R" << rotate_x << ", " << rotate_y << ", " << rotate_z << std::endl;
	}

	//	cout << "Mouse: T(" << translate_x << ", " << translate_y << ", " << translate_z << " )  R(" << rotate_x << ", " << rotate_y << ")\n";

	mouse_old_x = x;
	mouse_old_y = y;
}

