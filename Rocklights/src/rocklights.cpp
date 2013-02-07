/*
*	main.cpp
*	RockefellerCenter
*
*	Created	by Eitan Mendelowitz on	9/7/05.
* 
*
*/
#include <windows.h>		// Header File For Windows

#include "globals.h"
#include "debug.h"

#include <vector>

#include <iostream>
#include <fstream>
#include <string> 
using namespace	std;

#include <stdlib.h>
#include <time.h>

#ifdef _MYWINDOWS
#include <math.h>
#endif
#include <GLUT/glut.h>

#include "TextureLoader.h"

#include "profile.h"

#include "BasePixel.h"
#include "SubPixel.h"
#include "LightElement.h"
#include "LECoveStick.h"
#include "Pixel.h"
#include "IHoldAndFade.h"
#include "ICycle.h"
#include "IGeneric.h"
#include "Interpolators.h"
#include "LightFile.h"
#include "Panel.h"
#include "Panels.h"
#include "DataEnablerFile.h"
#include "PeopleStats.h"
#include "WorldStats.h"
#include "PersonStats.h"
#include "Dummies.h"
#include "Tracker.h"
#include "Pattern.h"
//#include "MCEmpty.h"
#include "MCSinglePixelDance.h"
#include "SoundHash.h"
#include "AvatarCreatorHash.h"
#include "AmbientCreatorHash.h"
#include "ArrangementHash.h"
#include "MCGeneric.h"
#include "StringSeq.h"
#include "Transformer.h"
#include "StringUtils.h"

float	rtri;						// Angle For The Triangle
float	rquad;						// Angle For The Quad

string curLogFilePath;
ofstream logStream;

float camX =-100.0f;
float camY = -100.0f;
float camZ = (5.5f * 2.54f)	+ 100;

float rotZ =  30.0f	* 3.14f	/ 180.0f; // 0 mean	looking	down x axis
float lookX	= cosf(30.0f * 3.14f / 180.0f );
float lookY	= sinf(30.0f * 3.14f / 180.0f);
float lookZ	= 0.0f;

TextureLoader *textureLoader;
glTexture logoTexture;
GLfloat  logoLeft;	 
GLfloat logoRight; 
GLfloat logoTop;	
GLfloat logoBot;	 
bool logoOn = true;

//int mouseX;
//int mouseY;

//float	camRotX	= -90;

int	frameCnt = 0;
int	startTime;
int	screenWidth;
int	screenHeight;

bool fullscreen;

bool dispText;
char textBuf[50];

#define PERSPECTIVEONLY 0
#define PERSPECTIVETOPDOWN 1
#define TOPDOWNONLY 2
#define DISTORTORIGIN 3
#define DISTORT1 4
#define DISTORT2 5
#define DISTORT3 6
#define LOGOLIST 1
int viewState;

bool setCoordTransform; // only gets check if topdown it already true
bool isControlDown;
// start up	with lights	off
bool forceLightsOn;

// light on	(if	not	already	on)	and	off	time in	24h	format
int	hourOn ;
int	minOn ;

int	hourOff;
int	minOff;

int	hourRestart;
int	minRestart;

bool isOnBeforeOff;


//World	*world;
//TargetTexture	*targetTexture;

//DataEnabler *dataEnabler;
//int dataEnablerMaxCnt;
DataEnablers *dataEnablers;
Panels *panels;

//#define testSize 2000	
//Pixel	*tmpPix[testSize];
//Interpolator *tmpFade[testSize];
//Interpolators	*interps; 
SoundHash *soundHash;
AvatarCreatorHash *avHash;
AmbientCreatorHash *ambHash;
ArrangementHash *arHash;

PeopleStats	*peopleStats; 
WorldStats *worldStats;
Dummies	*dummies;
Transformer *transformer;
bool mouseMoveCam =	true;


int	minUSecsPerFrame;
int	maxUSecsPerFrame;

int	deltaTime;
int	timeOfDayUpdateTime;

bool isRendering = true;
bool isUsingTracker;
bool isSendingDMX;

Tracker	*tracker = NULL;

vector<string> keybindings;


int	forcedPeopleCnt	= -1;

typedef	struct												// Create A	Structure
{
	GLubyte	*imageData;										// Image Data (Up To 32	Bits)
	GLuint	bpp;											// Image Color Depth In	Bits Per Pixel.
	GLuint	width;											// Image Width
	GLuint	height;											// Image Height
	GLuint	texID;											// Texture ID Used To Select A Texture
} TextureImage;												// Structure Name

TextureImage textures[1];									// Storage For One Texture

void displayText(float x, float y, float z, const char* s) {
	glColor3f(0.0f, 0.0f, 0.0f);
	glRasterPos3f(x, y, z);
	char c;
	int i = 0;
	while((c = s[i++])  != '\0') {
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, c);
	}
}


void InitGL	( GLvoid )	   // Create Some Everyday Functions
{

	glShadeModel(GL_FLAT);							// Enable Smooth Shading
	glClearColor(1.0f, 1.0f, 1.0f, 0.5f);				// Black Background
	glClearDepth(1.0f);									// Depth Buffer	Setup
	glDepthFunc(GL_LEQUAL);								// The Type	Of Depth Testing To	Do
	glEnable ( GL_COLOR_MATERIAL );
	glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
	glAlphaFunc(GL_GREATER,0.1f);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_FASTEST);
	glutSetCursor(GLUT_CURSOR_NONE);

	textureLoader = NULL;
	string logoFile = CProfile::theProfile->String("logoFile", "NONE");
	if(logoFile == "NONE") {
		logoOn = false;
	} else {
		logoFile = "imgs\\" + logoFile;
		textureLoader = new TextureLoader();
		if(textureLoader->LoadTextureFromDisk(logoFile.c_str(), &logoTexture) == TRUE) {
			glBindTexture(GL_TEXTURE_2D, logoTexture.TextureID);
			logoOn = true;
			timeStamp(); clog << "INFO	Done loading " << logoFile << "\n";
		} else {
			logoOn = false;
			timeStamp(); clog << "WARNING  Unable to load " << logoFile << "\n";
		}
	}


	
	/*
	tmpDe = new DataEnabler("", testSize	* 4	* 4);
	for(int i = 0; i	< testSize;	i ++) {
	LightElement *tmpLe;
	LightElement *tmpLe2;
	LightElement *tmpLe3;
	LightElement *tmpLe4;
	tmpLe = new LECoveStick(tmpDe->data, i	* 12);
	tmpLe2	= new LECoveStick(tmpDe->data, 3 + (i *	12));
	tmpLe3	= new LECoveStick(tmpDe->data,	6 +	(i * 12));
	tmpLe4	= new LECoveStick(tmpDe->data,	9 +	(i * 12));
	tmpPix[i] = new Pixel(tmpLe,tmpLe2,tmpLe3,tmpLe4);
	tmpPix[i]->setDims(0 +	(i * 2.25),	0, 2 + (i *	2.25), 4);
	tmpPix[i]->addColor(0,	255, 255,255);
	tmpPix[i]->addColor(1,	255, 0,0);
	tmpPix[i]->addColor(2,	0, 255,0);
	tmpPix[i]->addColor(3,	0, 0,255);
	tmpPix[i]->update();
	}

	interps = new Interpolators();
	for(int i = 0; i	< testSize;	i ++) {
	tmpFade[i]	= new IHoldAndFade(tmpPix[i], 255, 0, 0, 5000, 0, 0, 255, 5000);
	//	   interps->add(tmpFade[i]);
	}
	*/



}



void displayProjection(int offx, int offy, int width, int height) {
	glViewport(offx,offy,width, height);						// Reset The Current Viewport
	glTexEnvf(GL_TEXTURE_ENV,GL_TEXTURE_ENV_MODE,GL_REPLACE);
	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection	Matrix

	// Calculate The Aspect	Ratio Of The Window
	gluPerspective(90.0f,-(GLfloat)width/(GLfloat)height,0.1f,40.0f	* 12.0f	* 2.54);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview	Matrix
	glLoadIdentity();


	//	glTranslatef(200.0f, 0.0, 0.0);

	gluLookAt(camX-(3.9f * lookX), camY-lookY	- (3.9f	* lookY), camZ - (3.9f * lookZ), camX +	lookX, camY	+ lookY, camZ +	lookZ, 0, 0, 1); 
	//  world->display();

	//	dummies->display();

	glEnable(GL_DEPTH_TEST);							// Enables Depth Testing
	glEnable(GL_BLEND); 
	glEnable(GL_ALPHA_TEST);
	panels->display();
	peopleStats->display();
	glDisable(GL_ALPHA_TEST);
	glDisable(GL_BLEND);  
	glDisable(GL_DEPTH_TEST);							// Enables Depth Testing

	if(dispText) {
		MasterController *mc = MasterController::curMasterController;
		if(mc) {
			if(mc->name != "") {
				sprintf(textBuf, "State: %s", mc->name.c_str());
				displayText(500, -200, -100, textBuf);
				if( mc->timeToIntermission >= 0) {
				sprintf(textBuf, "Time: %i", mc->timeToIntermission);
				displayText(500, -200, -124, textBuf);
				sprintf(textBuf, "FPS: %.2f", ((float)	frameCnt * 1000.0f)	/ (float) (glutGet(GLUT_ELAPSED_TIME) - startTime));	
				displayText(500, -200, -148, textBuf);
				}
			}
		}
	}
	

//GLUT_BITMAP_TIMES_ROMAN_24

} 

void displayClear(int offx,	int	offy, int width, int height) {

	glClearColor(0.0f, 0.0f, 0.0f, 0.5f);				// Black Background


	glViewport(offx,offy,width, height);						// Reset The Current Viewport
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,width,height,0,-400,400);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glBegin(GL_QUADS);



	glColor3f(0.0f, 0.0f,	0.0f);
	//	 this is in	strange	world coords
	glVertex3f(0, height,0.0f);		
	glVertex3f(width, height,0.0f);			
	glVertex3f(width, 0, 0.0f);			
	glVertex3f(0, 0, 0.0f);		
	glEnd();					// Enable Texture Mapping

}


void displayLogo(int offx, int offy, int width,	int	height)	{
	if(! logoOn) return;

	int w = width-offx;
	int h = height-offy;
	glViewport(offx,offy,w,h);						// Reset The Current Viewport
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0,w,h,0,-100,100);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glCallList(LOGOLIST);

	glEnable(GL_TEXTURE_2D);
			glBegin(GL_QUADS);
	glColor3f(0.0f, 0.0f,	0.0f);
	//	 this is in	strange	world coords
	glTexCoord2f(0.0f, 0.0f);	glVertex3f(logoLeft, logoBot,0.0f);		
	glTexCoord2f(1.0f, 0.0f);	glVertex3f(logoRight, logoBot,0.0f);			
	glTexCoord2f(1.0f, 1.0f);glVertex3f(logoRight, logoTop, 0.0f);			
	glTexCoord2f(0.0f, 1.0f);glVertex3f(logoLeft, logoTop, 0.0f);		
	glEnd();					// Enable Texture Mapping
	glDisable(GL_TEXTURE_2D);

}

void displayTopDown(int	offx, int offy,	int	width, int height) {
	glViewport(offx,offy,width, height);						// Reset The Current Viewport
	/*

	glMatrixMode(GL_PROJECTION);						// Select The Projection Matrix
	glLoadIdentity();									// Reset The Projection	Matrix

	// Calculate The Aspect	Ratio Of The Window
	gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f	* 12.0f);

	glMatrixMode(GL_MODELVIEW);							// Select The Modelview	Matrix
	glLoadIdentity();									

	*/
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	//	glOrtho(-1000,1000,1000, -1000,-400,400);
	glOrtho(1000,-1000,-1000,	1000,-400,400);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glRotatef(90.0f, 0.0f, 0.0f, 1.0f);
	glRotatef(90.0f, 0.0f, 0.0f, 1.0f);

	int pt = -1;
	switch(viewState) {
		case DISTORTORIGIN:
			pt = 0;
			break;
		case DISTORT1:
			pt = 1;
			break;
		case DISTORT2:
			pt = 2;
			break;
		case DISTORT3:
			pt = 3;
			break;
	}


	
	panels->topDisplay();


	glBegin(GL_LINES);
	glColor3f(1.0f, 1.0f,	1.0f);		
	glVertex3f(camX ,camY,0.0f);		
	glColor3f(0.0f, 0.0f,	0.5f);		
	glVertex3f(camX +	lookX *	12,	camY + lookY * 12, 0.0f);		
	glEnd();
	
	if(pt >= 0) {
			glColor3f(1.0f, 0.0f,	1.0f);		
			glBegin(GL_LINE_LOOP);
			for(int i = 0; i < 4; i++) {
			glVertex2f(transformer->pt[i][0], transformer->pt[i][1]);	
			}
			glEnd();
			glPointSize(10.0);
			glBegin(GL_POINTS);
			glVertex2f(transformer->pt[pt][0], transformer->pt[pt][1]);	
			glEnd();
			glPointSize(3.0);
			glColor3f(0.0f, 0.0f,	0.0f);		
			glBegin(GL_POINTS);
			glVertex2f(transformer->pt[pt][0], transformer->pt[pt][1]);	
			glEnd();
	}

	peopleStats->display();


}

void render() {

	//int	renderWidth	= screenWidth *	.5f;
	glClear(GL_COLOR_BUFFER_BIT |	GL_DEPTH_BUFFER_BIT);	// Clear Screen	And	Depth Buffer

	switch(viewState) {
		case PERSPECTIVEONLY:
			displayProjection(-75,0, screenWidth, screenHeight);
			displayLogo(0,0, screenWidth , screenHeight);

			break;
		case PERSPECTIVETOPDOWN: {
			int	halfWidth =	screenWidth	* .5f;
			displayProjection(0,0, halfWidth , screenHeight);
			displayTopDown(halfWidth , 0, halfWidth, screenHeight); }
			break;
		case DISTORTORIGIN:
//			displayTopDown(-screenWidth * 1.5, -screenHeight * 1.5, screenWidth * 4, screenHeight * 4);
//			break;
		case DISTORT1:
		case DISTORT2:
		case DISTORT3:
		case TOPDOWNONLY:
			displayTopDown(- screenWidth * .6, -screenHeight * .15 , screenWidth * 1.75 , screenHeight * 1.75);
			break;
	}

	glutSwapBuffers (	);
}

void display ( void	)	// Create The Display Function
{
	glClear(GL_COLOR_BUFFER_BIT |	GL_DEPTH_BUFFER_BIT);	// Clear Screen	And	Depth Buffer
	glutSwapBuffers (	);

}


void frame(int t, int dt)	// Create The Display Function
{
	if(Globals::isOn) {

		peopleStats->update(t, dt);
		worldStats->update(t,	dt);
		if (forcedPeopleCnt >= 0)	{
			worldStats->peopleCnt =	forcedPeopleCnt;
		}
		MasterController::curMasterController->update(worldStats,	peopleStats, t,	dt);
		panels->update();
		if(isSendingDMX) {
			dataEnablers->sendDMX();
		}
		if(isRendering) {
			render();
		}
		dataEnablers->clear();
		frameCnt++;
	} 


}
void turnOn() {
	Globals::isOn =	true;
	glClear(GL_COLOR_BUFFER_BIT	| GL_DEPTH_BUFFER_BIT);	// Clear Screen	And	Depth Buffer
	glClearColor(1.0f, 1.0f, 1.0f, 0.5f);				// white Background
	glutSwapBuffers	( );

	timeStamp(); clog << "INFO  Turning on lights\n";

}
void turnOff() {
	Globals::isOn =	false;
	if(isSendingDMX) {
		dataEnablers->clear();
		// make	sure the lights	gets the clear
		dataEnablers->sendDMX();
		dataEnablers->sendDMX();
		dataEnablers->sendDMX();
		dataEnablers->sendDMX();
		dataEnablers->sendDMX();
	}


	if(isRendering)	{
		glClear(GL_COLOR_BUFFER_BIT	| GL_DEPTH_BUFFER_BIT);	// Clear Screen	And	Depth Buffer
		displayClear(0,0, screenWidth, screenHeight);  
		glutSwapBuffers	( );
	}


	timeStamp(); clog << "INFO  Turning off lights\n";
}



void reshape ( int width , int height )	  // Create	The	Reshape	Function (the viewport)
{
	if (height==0)										// Prevent A Divide	By Zero	By
	{
		height=1;										// Making Height Equal One
	}
	screenWidth =	width;
	screenHeight = height;

	frameCnt =0;
	startTime =	glutGet(GLUT_ELAPSED_TIME);

}
#include <direct.h>

string GetCurrentDir() 
{ 
	char buf[200];
	getcwd(buf, 200);
	string s = buf;
	return s;
} 

string getExeName(string strFullPathToExe)
{
	int	Position = strFullPathToExe.find_last_of("\\");
	strFullPathToExe = strFullPathToExe.erase(0, Position +1);

	return strFullPathToExe;
}
// helper from http://goff.nu/techarticles/development/cpp/createprocess.html
int	ExecuteProcess(string &FullPathToExe, string &Parameters, int SecondsToWait)
{
	int	iMyCounter = 0,	iReturnVal = 0;
	DWORD dwExitCode;

	/* - NOTE -	You	could check	here to	see	if the exe even	exists */

	/* Add a space to the beginning	of the Parameters */
	if (Parameters.size() != 0 )
	{
		Parameters.insert(0," ");
	}

	/* When	using CreateProcess	the	first parameter	needs to be	the	exe	itself */
	Parameters = getExeName(FullPathToExe).append(Parameters);

	/*
	The	second parameter to	CreateProcess can not be anything but a	char !!
	Since we are wrapping this C function with strings,	we will	create
	the	needed memory to hold the parameters
	*/

	/* Dynamic Char	*/
	char * pszParam	= new char[Parameters.size() + 1];

	/* Verify memory availability */
	if (pszParam ==	0)
	{
		/* Unable to obtain	(allocate) memory */
		return 1;
	}
	const char*	pchrTemp = Parameters.c_str();
	strcpy(pszParam, pchrTemp);

	/* CreateProcess API initialization	*/
	STARTUPINFO	siStartupInfo;
	PROCESS_INFORMATION	piProcessInfo;
	memset(&siStartupInfo, 0, sizeof(siStartupInfo));
	memset(&piProcessInfo, 0, sizeof(piProcessInfo));
	siStartupInfo.cb = sizeof(siStartupInfo);

	/* Execute */
	if (CreateProcess(FullPathToExe.c_str(), pszParam, 0, 0, false,
		CREATE_DEFAULT_ERROR_MODE, 0, 0, &siStartupInfo,
		&piProcessInfo)	!= false)
	{
		/* A	loop to	watch the process. Dismissed with SecondsToWait	set	to 0 */
		GetExitCodeProcess(piProcessInfo.hProcess, &dwExitCode);

		while (dwExitCode ==	STILL_ACTIVE &&	SecondsToWait != 0)
		{
			GetExitCodeProcess(piProcessInfo.hProcess, &dwExitCode);
			Sleep(500);
			iMyCounter += 500;

			if (iMyCounter >	(SecondsToWait * 1000))
			{
				dwExitCode =	0;
			}
		}
	}
	else
	{
		/* CreateProcess failed. You could also	set	the	return to GetLastError() */
		iReturnVal = 2;
	}

	/* Release handles */
	CloseHandle(piProcessInfo.hProcess);
	CloseHandle(piProcessInfo.hThread);

	/* Free	memory */
	delete[]pszParam;
	pszParam = 0;

	return iReturnVal;
}


void cleanup() {
	cout << "starting cleanup" << endl;

	dataEnablers->clear();
	dataEnablers->sendDMX();
	dataEnablers->sendDMX();
	dataEnablers->sendDMX();
	delete dataEnablers;
	cout << "dataEnablers deleted" << endl;
	//	delete panels;
	delete soundHash;
	cout << "soundHash deleted" << endl;
	delete avHash;
	cout << "avHash deleted" << endl;
	delete  ambHash;
	cout << "ambHash deleted" << endl;
	delete arHash;
	cout << "arHash deleted" << endl;
	delete peopleStats;
	cout << "peopleStats deleted" << endl;
	delete worldStats;
	cout << "worldStats deleted" << endl;
	delete dummies;
	cout << "dummies deleted" << endl;
	if(textureLoader) {
		textureLoader->FreeTexture(&logoTexture);
		delete textureLoader;
		cout << "textureLoader deleted" << endl;
	}
	StringSeq::destroy();
	cout << "sequence stores deleted" << endl;

	if(CProfile::theProfile->Bool("deleteTracker", false)) {
		cout << "Deleting Tracker";
		if (tracker)	delete tracker;
		cout << "Tracker Deleted";
	} else {
		if(tracker) {
			cout << "SKIPPING TRACKER DELETION" << endl;
			timeStamp(); clog << "INFO  SKIPPING TRACKER DELETION\n";
		}
	}
	timeStamp(); clog	<< "INFO  Done cleaning	proceeding with	exit\n";
	 cout	<< " Done cleaning	proceeding with exit\n";
	clog.flush();
	cout << "Done flushing log" << endl;
	logStream.close();
	cout << "Done closeing stream" << endl;


}

void quit()	{
	cleanup();
	cout << "Done with cleanup" << endl;
	exit(0);
//		_exit(0);

}

void restart() {

	timeStamp(); clog << "INFO restart called\n";

	
	string restartBat =	GetCurrentDir() + "\\" + CProfile::theProfile->String("restartBatFile", "");
	string restartArgs = GetCurrentDir() + "\\" + CProfile::theProfile->String("executablePath",	"");

	if (ExecuteProcess(restartBat, restartArgs,	0) != 0) {
		timeStamp(); clog << "ERROR	 unable	to call	restart	bat	file" << endl;
				Globals::hasError = true;

	}
	timeStamp(); clog << "INFO restart batfile started\n";

	cleanup();
	string ftpBat	= CProfile::theProfile->String("ftpBatFile", "");

	if (ExecuteProcess(ftpBat, curLogFilePath, 0) != 0)	{
		timeStamp(); clog << "ERROR	 unable	to ftp log file" <<	endl;

	}

	exit(0);
//	_exit(0);

}

void keyUpFunc ( unsigned char key,	int	x, int y ) {
	

	switch ( key ) {
  case '=':
	  forcedPeopleCnt	+=1;
	  // keep	this to	cou	t since	it is for keyboard interaction
	  cout <<	"forced	people cnt " <<	forcedPeopleCnt	<< "\n";
	  break;
  case '-':
	  forcedPeopleCnt	-= 1;
	  // keep	this to	co ut also since it	is for keyboard	interaction
	  cout <<	"forced	people cnt " <<	forcedPeopleCnt	<< "\n";
	  break;
	  /*
	  case 's' : {// scale the room
	  PersonStats	*se	= peopleStats->getSE();	
	  if (se != NULL)	{
	  BasePixel	*pix =	Panels::thePanels->panels[Panels::A].getPixel(16, 9);
	  tracker->scaleX =	se->x/pix->x;
	  tracker->scaleY =	se->y/pix->y;
	  }}
	  cout <<	"scaleX	= "	<< tracker->scaleX << endl;
	  cout <<	"scaleY	= "	<< tracker->scaleY << endl;

	  break;
	  case 'S' : {
	  tracker->scaleX	= 1.0f;
	  tracker->scaleY	= 1.0f;
	  }
	  break;
	  */
  case '0':
	  mouseMoveCam = ! dummies->setSelectMode(-1);
	  break;
  case '1':
	  mouseMoveCam = ! dummies->setSelectMode(0);
	  break;
  case '2':
	  mouseMoveCam = ! dummies->setSelectMode(1);
	  break;
  case '3':
	  mouseMoveCam = ! dummies->setSelectMode(2);
	  break;
  case '4':
	  mouseMoveCam = ! dummies->setSelectMode(3);
	  break;
  case '5':
	  mouseMoveCam = ! dummies->setSelectMode(4);
	  break;
  case '6':
	  mouseMoveCam = ! dummies->setSelectMode(5);
	  break;
  case '7':
	  mouseMoveCam = ! dummies->setSelectMode(6);
	  break;
  case '8':
	  mouseMoveCam = ! dummies->setSelectMode(7);
	  break;
  case '9':
	  mouseMoveCam = ! dummies->setSelectMode(8);
	  break;
  case '!':
	  mouseMoveCam = ! dummies->setSelectMode(9);
	  break;
  case '@':
	  mouseMoveCam = ! dummies->setSelectMode(10);
	  break;
  case '#':
	  mouseMoveCam = ! dummies->setSelectMode(11);
	  break;
  case '$':
	  mouseMoveCam = ! dummies->setSelectMode(12);
	  break;
  case '%':
	  mouseMoveCam = ! dummies->setSelectMode(13);
	  break;
  case '^':
	  mouseMoveCam = ! dummies->setSelectMode(14);
	  break;
  case '&':
	  mouseMoveCam = ! dummies->setSelectMode(15);
	  break;
  case '*':
	  mouseMoveCam = ! dummies->setSelectMode(16);
	  break;
  case '(':
	  mouseMoveCam = ! dummies->setSelectMode(17);
	  break;
  case 'f':
	  fullscreen = ! fullscreen;
	  if(fullscreen) {
		  glutFullScreen (	); // Go Into Full Screen Mode
	  }	else {
		  glutReshapeWindow	 ( 1000, 500 );	
		  glutPositionWindow(4,	29);
	  }
	  break;
  case 'l':
	  logoOn = ! logoOn;
	  break;
  case 'c':
	  Globals::displayCoord = ! Globals::displayCoord;
	  break;
  case 't':
	  	glDisable(GL_POINT_SMOOTH);

	  switch(viewState) {
		case PERSPECTIVEONLY:
			viewState = PERSPECTIVETOPDOWN;
			break;
		case PERSPECTIVETOPDOWN: 
			viewState = TOPDOWNONLY;
			break;
		case DISTORTORIGIN:
		case DISTORT1:
		case DISTORT2:
		case DISTORT3:
		case TOPDOWNONLY:
		default:
			transformer->calc();
					for(int i = 0; i < 5; i++) {
			dummies->setSelectMode(i);
			dummies->update(0,0);
		}

			viewState = PERSPECTIVEONLY;
			break;
	  }
	  break;
  case'u':
	  transformer->calc();
	  		for(int i = 0; i < 5; i++) {
			dummies->setSelectMode(i);
			dummies->update(0,0);
		}

	  break;
  case'D':
	  transformer->reset();
	   transformer->calc();
	  		for(int i = 0; i < 5; i++) {
			dummies->setSelectMode(i);
			dummies->update(0,0);
		}
  case 'd':
	glEnable(GL_POINT_SMOOTH);

	  switch(viewState) {
		case DISTORTORIGIN:
			viewState = DISTORT1;
			cout << "Adjust NW" << endl;
			break;
		case DISTORT1:
			cout << "Adjust SW" << endl;
			viewState = DISTORT2;
			break;
		case DISTORT2:
			viewState = DISTORT3;
			cout << "Adjust SE" << endl;
			break;
		case DISTORT3:
			cout << "Adjust NE" << endl;
			viewState = DISTORTORIGIN;
			break;
		default:
			cout << "Adjust NE" << endl;
			viewState = DISTORTORIGIN;
			break;
	  }
	  break;
  case 'F':	{
	  int curTime =	glutGet(GLUT_ELAPSED_TIME);

	  //	  int mod =	glutGetModifiers();
	  //	  if (mod == GLUT_ACTIVE_ALT)
	  // keep this cout
	  std::cout<<"FPS: "<< ((float)	frameCnt * 1000.0f)	/ (float) (curTime - startTime)	 <<std::endl;  }
			frameCnt =0;
			startTime =	Globals::curTime;

			break;
  case 'r':	{
	  isRendering	= !	isRendering;
			}
			break;

  case 'w':		 //	When Escape	Is Pressed...
	  timeStamp(); clog << "INFO	User initiated quit	with restart\n";
	  restart();
	  break;		// Ready For Next Case
  case 'q':
	  timeStamp(); clog << "INFO	User initiated quit	without	restart\n";
	  quit();
	  break;
  default:	
	  for(unsigned int i = 0; i < keybindings.size(); i +=2) {
		  if(key == keybindings[i][0]) {
			  new MCGeneric(peopleStats, keybindings[i+1], MasterController::curMasterController->name);
			  timeStamp(); clog << "INFO    User initiated state change (via keyboard)\n";
		  }
	  }

	  break;
	}
}

int	oldX = -1;
int	oldY = -1;

float dummyX;
float dummyY;
bool mouseRot =	false;
void mouse(int button, int state, int x, int y)	{
	if(state == GLUT_DOWN) {
		oldX = x;
		oldY = y;
	}	else {
		oldX = -1;
		oldY = -1;
	}

	mouseRot = button	== GLUT_LEFT_BUTTON;
}

void mouseMove(int x, int y) {
	if ((oldX	!= -1) && (oldY	!= -1))	{
		if (mouseRot) {
			if(mouseMoveCam) {
				camX -= ((float) (oldX - x)) * 10.0f;
				camY -= ((float) (oldY - y)) * 10.0f;
			}	else {
				dummyX = ((float) (x- oldX)) * 5.0f;
				dummyY = ((float) (y - oldY)) *	5.0f;
				dummies->update(dummyX,dummyY);
			}

		} else {
			//	  camRotY += ((float) (x - oldX)) *	1.0f;
			rotZ += ((float) (y -	oldY)) * .05f;
			lookX	= cosf(rotZ);
			lookY	= sinf(rotZ);
			normalize(lookX, lookY, lookZ);
		}
	}
	oldX = x;
	oldY = y;
}

void mouseEnterExit(int	state) {
	oldX = -1;
	oldY = -1;
}

void arrow_keys	( int a_keys, int x, int y ) { // Create	Special	Function (required for arrow keys)
	bool distort = true;
	float amt = 1.0f;
	int pt = 0;
	int dim =0;

	switch(viewState) {
		case DISTORTORIGIN:
			distort = true;
			break;
		case DISTORT1:
			pt = 1;
			break;
		case DISTORT2:
			pt = 2;
			break;
		case DISTORT3:
			pt = 3;
			break;
		default:
			distort = false;
			switch ( a_keys )	{
				case GLUT_KEY_UP:	  // When Up Arrow Is Pressed...
					camY -= .05f;
					break;
				case GLUT_KEY_DOWN:				  // When Down Arrow Is	Pressed...
					camY += .05f;
					break;
			}

	}
	if (distort) {
		switch ( a_keys )	{
			case GLUT_KEY_UP:
				amt *= -1 ;
				dim = 1;
			break;
			case GLUT_KEY_DOWN:
				dim = 1;
			break;
			case GLUT_KEY_LEFT:
				amt *= -1 ;
			break;
			case GLUT_KEY_RIGHT:
			break;
			}
		int mod = glutGetModifiers();
		if(mod ==GLUT_ACTIVE_SHIFT) {
			amt *= .25f;
		}
		transformer->incPoint(pt, dim, amt);
//		transformer->calc();
	}
	
}

void timerCallback (int	value) {
	

	int newTime =	glutGet(GLUT_ELAPSED_TIME);
	if (newTime >= Globals::curTime) { //	if time	wrapped	around ignore and use last delta
		deltaTime =	newTime	- Globals::curTime;
	}

	Globals::curTime = newTime;

	
	if(Globals::hasError) { // if there was somehting bad push it out
		clog.flush();
		string ftpBat	= CProfile::theProfile->String("ftpBatFile", "");
		if (ExecuteProcess(ftpBat, curLogFilePath, 0) != 0)	{
			timeStamp(); clog << "ERROR	 unable	to ftp log file" <<	endl;
		}
		Globals::hasError = false;
	}

	if (newTime >	timeOfDayUpdateTime) {
		updateTimeStamp();
		//FIX
		timeOfDayUpdateTime =	newTime	+ 60000; //	update onces a minute

		if (forceLightsOn) {
			if(! Globals::isOn) {
				turnOn();
			}
		}	else {
			if (isOnBeforeOff) {
				if(Globals::isOn)	{
					if(isBefore(hourOff, minOff, 
						Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min))	{
							turnOff();
						}	else if(isBefore(Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min,
							hourOn, minOn)) {
								turnOff();
							}
				}	else {
					if (isBefore(hourOn, minOn, 
						Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min) &&
						isBefore(Globals::timeOfDay->tm_hour,	Globals::timeOfDay->tm_min,
						hourOff,	minOff)) {
							turnOn();
						}
				}

			}	else { 
				if(Globals::isOn)	{ // eg	 on	at 7am off at 1am 
					if(isBefore(hourOff, minOff, Globals::timeOfDay->tm_hour,	Globals::timeOfDay->tm_min)	&&
						isBefore(Globals::timeOfDay->tm_hour,	Globals::timeOfDay->tm_min,	hourOn,	minOn))	{
							turnOff();
						}



				}	else {
					if(isBefore(Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min,
						hourOff, minOff))	{
							turnOn();
						}	else if(isBefore(hourOn, minOn,	 Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min)) {
							turnOn();
						}


				}
			}
		}

	}

	if(! Globals::isOn) {	// only	check for restart if off;
		if(hourRestart ==	Globals::timeOfDay->tm_hour) {
			if ((minRestart == Globals::timeOfDay->tm_min) ||
				(minRestart == Globals::timeOfDay->tm_min	- 1)) // minute	window
			{
				restart();
			}
		}
	}



	frame(Globals::curTime, deltaTime);

	if (isUsingTracker) {
		tracker->processTrackData(); // won't	do anything	if no new data
	}

	newTime =	minUSecsPerFrame - (glutGet(GLUT_ELAPSED_TIME) - Globals::curTime);	// see how we should wait until	next frame
	newTime =	(newTime <=	0) ? 0 : newTime;


	bool newGrab = true;

	if (isUsingTracker) {
		newGrab	= tracker->grab(newTime);
	}

	if ((newTime > 0)	&& newGrab)	{ // there might be	extra time
		newTime	= minUSecsPerFrame - (glutGet(GLUT_ELAPSED_TIME) - Globals::curTime); // see how we	should wait	until next frame
		glutTimerFunc (newTime,	timerCallback, value);
	}	else { // no time loop right away
		glutTimerFunc (0, timerCallback, value);
	}

}

#include "Bounds.h"

int	main ( int argc, char**	argv )	 //	Create Main	Function For Bringing It All Together
{
	updateTimeStamp();
	seedRandom();
	isControlDown =	false;
	CProfile*	prof = new CProfile();
	
	if (!	prof->Load("profile.txt")) {
		timeStamp(); clog << "ERROR	 Profile file (profile.txt)	failed to load"	<< endl;
		Globals::hasError = true;
	}

	dispText = prof->Bool("displayText", false);


	int m = Globals::timeOfDay->tm_mon	+ 1;
	int d = Globals::timeOfDay->tm_mday;
	int y = Globals::timeOfDay->tm_year + 1900;

	char fileName[19];
	sprintf(fileName, "%02i_%02i_%i.log",m,d,y);

//	string	logFileDir = CProfile::theProfile->String("logFileDir",	"");
	string logFileDir = GetCurrentDir() + "\\logs\\";


	curLogFilePath = logFileDir+fileName;

	logStream.open(curLogFilePath.c_str(), ios::out	| ios::app);
	if (logStream.is_open()) {
		clog.rdbuf(logStream.rdbuf());
	}	else {
		cerr << "ERROR: unable to	open logfile" << endl;
				Globals::hasError = true;

	}

	std::clog.fill('0');


	//first delete	last months	file
	if	(m == 1) {
		sprintf(fileName, "logs/12_%02i_%i.log", d,y);
	} else	{
		sprintf(fileName, "logs/%02i_%02i_%i.log", m - 1,d,y);
	}
	if( remove(fileName) == -1	) {
		timeStamp(); clog << "INFO unable to delete	log	file " << fileName << "\n";
	} else	{
		timeStamp(); clog << "INFO successfully	deleted	log	file " << fileName << "\n";
	}
	if	(d == 30) {	// also	delete 31st	of last	month
		if	(m == 1) {
			sprintf(fileName, "logs/12_31_%i.log",	y);
		} else	{
			sprintf(fileName, "logs/%02i_31_%i.log", m - 1,y);
		}
		if( remove(fileName) != -1	) {
			timeStamp(); clog << "INFO successfully	deleted	log	file " << fileName << "\n";
		}

	}
	/*
	string fileName =	"log_";
	clog << fileName << endl;
	int tmp =	Globals::timeOfDay->tm_mon + 1;
	if ( tmp < 10) fileName +=  "0";
	fileName = fileName +	tmp	+ "_";
	clog << fileName << endl;
	tmp =	Globals::timeOfDay->tm_mday;
	clog << fileName << endl;
	if ( tmp < 10) fileName +=  "0";
	clog << fileName << endl;
	fileName += tmp +	"_";
	clog << fileName << endl;
	tmp =	Globals::timeOfDay->tm_year	+ 1900;
	clog << fileName << endl;
	if ( tmp < 10) fileName +=  "0";
	clog << fileName << endl;
	fileName += tmp +	".txt";
	clog << fileName << endl;
	*/
	DEBUGMSG();

	timeOfDayUpdateTime =	0;

	time_t tim=time(NULL);

	timeStamp(); clog	<< "INFO  ------------------------------------------------\n";
	timeStamp(); clog	<< "INFO  "<<  ctime(&tim);
	timeStamp(); clog	<< "INFO  ------------------------------------------------\n";
	timeStamp(); clog	<< "INFO  Starting up."	<< endl;

	string ftpBat	= CProfile::theProfile->String("ftpBatFile", "");

	if (ExecuteProcess(ftpBat, curLogFilePath, 0) != 0)	{
		timeStamp(); clog << "ERROR	 unable	to ftp log file" <<	endl;
				Globals::hasError = true;

	}


	//			printf("Date is	%d/%02d/%02d\n", Globals::timeOfDay->tm_year+1900, Globals::timeOfDay->tm_mon+1, Globals::timeOfDay->tm_mday);
	//		   printf("Time	is %02d:%02d\n", Globals::timeOfDay->tm_hour, Globals::timeOfDay->tm_min);


	// start up with lights off

	forceLightsOn	= prof->Bool("forceLightsOn", false);


	hourRestart =	prof->Int("hourRestart", 2);
	minRestart = prof->Int("minRestart", 30);

	hourOn = prof->Int("hourOn",7);
	minOn	= prof->Int("minOn",0);

	hourOff =	prof->Int("hourOff",23);
	minOff = prof->Int("minOff",59);

	isOnBeforeOff	= isBefore(hourOn, minOn, hourOff, minOff);

	logoLeft = prof->Float("logoLeft", 0.0f);
	logoTop =	prof->Float("logoTop", 0.0f);
	logoRight	= prof->Float("logoRight", 100.0f);
	logoBot =	prof->Float("logoBot", 50.0f);

	Globals::displayCoord = prof->Bool("displayCoords", false);
	fullscreen = prof->Bool("fullscreen",	true);
	if(prof->Bool("topdown", false)) {
		viewState = PERSPECTIVETOPDOWN;
	} else {
		viewState = PERSPECTIVEONLY	;
	}

	BasePixel::initBlank();

	isRendering =	prof->Bool("render", true);

	isUsingTracker = prof->Bool("useTracking", false);
	isSendingDMX = prof->Bool("sendDMX", false);


	normalize(lookX, lookY, lookZ);

	dataEnablers = new DataEnablers();


	float	lightScale = prof->Float("lightScale", 2.54f);

	//	dataEnablers->setupEnablers(const_cast<char	*>(prof->String("datnablersFile", "foo.txt")));
	dataEnablers->setupEnablers(const_cast<char *>(prof->String("dataEnablersFile", "dataEnablers.txt")));

	LightFile	*rgbLightFile =	new	LightFile(const_cast<char *>(prof->String("lightsFile",	"lights.txt")));
	rgbLightFile->setScale(lightScale);
	rgbLightFile->readWidthHeight();

	LightFile	*targetLightFile = new LightFile(const_cast<char *>(prof->String("targetFile", "targets.txt")));
	targetLightFile->setScale(lightScale);
	targetLightFile->readWidthHeight();


	{// extra	paren to make windows happy
		for(int i	= 0; i < Panels::PANEL_CNT;	i++) {
			targetLightFile->setMinCol(i, rgbLightFile->getMinCol(i)) ;
			targetLightFile->setMinRow(i, rgbLightFile->getMinRow(i)) ;
			targetLightFile->setMaxCol(i, rgbLightFile->getMaxCol(i)) ;
			targetLightFile->setMaxRow(i, rgbLightFile->getMaxRow(i)) ;
		}}

	Pattern::theCurPattern = new Pattern();

	panels = new Panels(prof->Float("lightWidth",	.2)	* .5f *	lightScale,	prof->Float("lightHeight", .4) * .5f * lightScale, prof->Float("targetRadius", .4) * lightScale	);

	panels->setLights(rgbLightFile, dataEnablers->dataEnabler, false);
	panels->setLights(targetLightFile, dataEnablers->dataEnabler,	true);

	panels->panels[0].calcStats();
	panels->panels[1].calcStats();

	soundHash = new SoundHash();

	avHash = new AvatarCreatorHash();
	ambHash = new AmbientCreatorHash();
	arHash = new ArrangementHash();

	BasePixel *psw = panels->panels[Panels::A].getPixel(17,11);

		transformer = new Transformer(psw->x, psw->y); // location of A,17,10



	peopleStats =	new	PeopleStats(512); 
	worldStats = new WorldStats();
//	new MCEmpty(peopleStats);
	new MCGeneric(peopleStats, "Empty", "NONE");
	//	personStats	= new PersonStats(0, 0,	10,	6, 6);
	dummies =	new	Dummies(peopleStats, worldStats);
	dummies->setRoomDim(panels->panels[Panels::A].maxX, panels->panels[Panels::A].maxY);

	string tmp = prof->String("keybindings", "");
	StringUtils::split(tmp, keybindings);


	if(isUsingTracker) {
		tracker	= new Tracker(peopleStats, worldStats);
		timeStamp(); clog << "INFO	Attaching to moderator " <<	prof->String("moderatorIP",	"localhost") <<	"\n";
		isUsingTracker = tracker->init(const_cast<char *>(prof->String("moderatorIP", "localhost")));
	}	else {
		timeStamp()	; clog << "INFO	 Not using tracking	(due to	setting	in profile.txt)\n";
	}


	mouseMoveCam = ! dummies->setSelectMode(0);


	minUSecsPerFrame = (int) (1000.0f	/ prof->Float("maxFPS",	32.0f));
	maxUSecsPerFrame =(int) (1000.0f / prof->Float("minFPS", 24.0f));
	//	std::cou t << "Frame delay between " <<	minUSecsPerFrame / 1000.0f << "	and	" << maxUSecsPerFrame /	1000.0f	<< "secs" << std::endl;

	glutInit			  (	&argc, argv	); // Erm Just Write It	=)
	glutInitDisplayMode (	GLUT_RGBA |	GLUT_DOUBLE	| GLUT_DEPTH); // Display Mode
	glutInitWindowSize  (	1000, 500 ); //	If glutFullScreen wasn't called	this is	the	window size
	glutCreateWindow	  (	"Electroland" ); //	Window Title (argv[0] for current directory	as title)
	//	glutFullScreen		( );		  // Put Into Full Screen
	InitGL ();
	glutDisplayFunc	  (	display	);	// Matching	Earlier	Functions To Their Counterparts
	glutReshapeFunc	  (	reshape	);
	glutKeyboardFunc	  (	keyUpFunc );
	glutSpecialFunc	  (	arrow_keys );
	glutEntryFunc		(mouseEnterExit);
	glutMouseFunc(mouse);
	glutMotionFunc (mouseMove);

	//  world	= new World(prof);
	// targetTexture = new TargetTexture("target2.tga");
	Globals::curTime = glutGet(GLUT_ELAPSED_TIME);
	startTime	= Globals::curTime;

	if(fullscreen) {
		glutFullScreen (	); // Go Into Full Screen Mode

	}
	turnOff();


	timerCallback(0);
	//	displayClear(0,0, screenWidth, screenHeight);


	/*
	//LIGHT_CEILNG
	for(int i = 0; i	< LIGHT_CNT_W; i++)	{
	int rowOffset = i * LIGHT_CNT_W;
	for(int j = 0;	j <	LIGHT_CNT_H; j++) {
	Lights[rowOffset	+ j].set(i,	 -9.0f + (float) i,	1.0f, -1.25	* j, 1.0f, 0.0f);
	}
	}
	*/
	//  Lights[LIGHT_CNT - 1]	= new Light(LIGHT_CNT -1, LIGHT_WALL , -1.0f, .5f, (-1.25*(LIGHT_CNT - 2)) - .5, 1.0f, 0.0f);
	//  Lights[LIGHT_CNT - 1]->setColor(0.0, 1.0f, 1.0f);
	glutMainLoop		  (	);			// Initialize The Main Loop

	delete prof; 
	delete rgbLightFile;
	delete targetLightFile;

	return 0;
}

