//UNDONE REMOVE THIS FOR VC6


#ifndef __GLOBALS_H__
#define __GLOBALS_H__

#define _MYWINDOWS
//#define _MYDEBUG


// number of dummys in array for testing/gui/mouse
#define DUMMYCNT 100

#define PEOPLERENDERHEIGHT 170

//calculating and x,y positions ceiling row and col is precomputed down to a fixed accuracy relative to the 
// mesurement units.  
#define SCALEFACTOR 10

// max number of avatars allowd to be in an individual group (pattern) for a particular person
#define MAXAVATARS 10
// number of differnt groups (patterns) allowed for a person
#define AVATARGRPS 2

// max number of simultanious wave
#define MAXWAVES 10

// the size "people" should be drawn on the screen
#define PERSONSTATSIZE 35

#define MAXOFFSETPIXELSPERAVATAR 100

#define TYZXATTACHATTEMPTS 2

// ((20 * 12) + (6 * 7) + (8 * 7) ) * 4 = (240 + 42 + 55) * 4 = 
#define STICKCNT 1352

#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <time.h>
#include <iostream>
#include <iomanip>
#include <string>
#include <GLUT/glut.h>


using namespace std;



inline void seedRandom() {
	srand(time(NULL));
}
inline int random(int max) {
	if(max == 0) return 0;
  return rand()%max;
}


inline void normalize(float &x, float &y, float &z) { 
  float scale = 1.0f / sqrtf((x * x) + (y * y) + (z * z));
  x *= scale;
  y *= scale;
  z *= scale;
}


class Globals {
public:
  static int curTime;
  static tm *timeOfDay;
  static bool hasError;
  static bool isOn;
  static bool displayCoord;
}
;

inline void updateTimeStamp() {
	time_t tm = time(NULL);
	Globals::timeOfDay = localtime(&tm);
}


inline bool isBefore(int h1, int m1, int h2, int m2) {
	if ((h1 <h2) || ((h1 == h2) && (m1 < m2))) return true;
	return false;	
}

inline void timeStamp() {
	clog << setw(2) << Globals::timeOfDay->tm_hour << ":" << 
			setw(2) << Globals::timeOfDay->tm_min << "   " ;
}
;

inline void dateStamp() {
	clog << Globals::timeOfDay->tm_mon+1 << "/" << 
			Globals::timeOfDay->tm_mday << "/" <<
			Globals::timeOfDay->tm_year+1900 << " ";
}
;




//			printf("Date is %d/%02d/%02d\n", Globals::timeOfDay->tm_year+1900, Globals::timeOfDay->tm_mon+1, Globals::timeOfDay->tm_mday);
 //            printf("Time is %02d:%02d\n", );

#endif