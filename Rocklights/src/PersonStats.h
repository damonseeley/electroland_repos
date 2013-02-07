/*
 *  PersonStats.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __PERSONSTATS_H__
#define __PERSONSTATS_H__
#include "globals.h"

#include "profile.h"

#include "Interpolators.h"
#include "Avatar.h"
#include "PeopleStats.h"
#include <GLUT/glut.h>

#include <iostream>
using namespace std;

class PeopleStats;

class Avatar ;

class Interpolators;

struct AvatarGroup {
  Avatar  *avatar[MAXAVATARS];
  int cnt;
  float scale;
  bool isActive;

}
;
class PersonStats {
	static int nextColor;
  AvatarGroup avatarGroups[AVATARGRPS];


public:
  static float pixelStick;
  static float pilDistSquaredEnter;
  static float pilDistSquaredExit;

  int nearPillar;

  bool exited;
  Interpolators *inpterps;
	PeopleStats *hashtable;
  int hashcode;
  char charBuf[30];

  // next within a hash bucket
  PersonStats *next;
  PersonStats *prev;

  // next in the whole world
  PersonStats *nextTot;
  PersonStats *prevTot;

  float renderSize;

  unsigned long id;
  
  float x;
  float y;
  float h;
  
  float lastX;
  float lastY;
  float lastH;
  
  float dX;
  float dY;
  float dH;


  int color;

  float r;
  float g;
  float b;

  int enterTime;

  int col;
  int row;
  

//  float perNE;
//  float perNW;
//  float perSE;
//  float perSW;

//  float lastPercentSouth;
 // float lastPercentEast;
 // float percentSouth;
 // float percentEast;

  BasePixel *pixel ;
 
  bool wasUpdated;
  bool inited;
  
  PersonStats(unsigned long id, int curTime);
  PersonStats() ;
  ~PersonStats();

  void update(float x, float y, float z);
  
  void update(int curTime, int deltaTime);

  void setPeopleStats(PeopleStats *hash); 

  void println() { cout << id << ":" << " (" << x << ", " << y  << ", " <<h << ")" << endl; }


  void setColor(float fr, float fg, float fb) { r = fr; g= fg; b = fb;}

  void display();

  
  void setAvatarGroupActivation(int i, bool b);
  void setAvatarGroupScale(int i, float s);

  void addAvatar(Avatar  *av, int g);

  void exitAvatars();
  void clearAndDestroyAvatars(int i);

  void setRenderSize(float f) { renderSize = f; }

// return 0 if not near any
// returns negative value if near back wall (value is negative closest row)
// return positive number for c-j
  void calcNearPillar();
  float distSquaredToPillar(int pil) ;
  void displayText(float x, float y, float z, const char* s);
}

;
#endif