#ifndef __TRACKER_H__
#define __TRACKER_H__

#include "globals.h"
#include "debug.h"

#include <iostream>
using namespace std;

#include "PeopleStats.h"
#include "PersonStats.h"
#include "WorldStats.h"
#include "PersonTrackAPI.h"
#include "Pattern.h"
#include "Bounds.h"


class Tracker {
  PersonTrackAPI *trax;
  PeopleStats *peopleStats;
  WorldStats *worldStats;
  Bounds *bounds;

  int cameraCnt;

  bool hasNewData;

  void checkCameras();
  void checkEnterExits();
  void checkPeople();


public:
	
  float scaleX;
  float scaleY;

  float offsetX;
  float offsetY;

  float cutoffX;
  
  float excludeBoxE;
  float excludeBoxN;
  float excludeBoxS;

  bool useOldCoordCalibration;

  int reportModeratorErrorTime;
  int reportModeratorEmptyRoomErrorTime;
  int reportModeratorEmptyRoomErrorInterval;
  bool isModeratorDown;

  Tracker(PeopleStats *ps, WorldStats *ws);
  ~Tracker();
  void clearTrackingData();
  bool init(char *modIP);
  bool grab(int sleepTime);
  void processTrackData();
  // will attempt to grab in alloted time
  // if time will return false
  // else will return true

  void destroy() { delete trax; }

  static bool errorMsg(const char *msg) { clog << "ERROR: From Trax " << msg << "/n"; 		Globals::hasError = true;
return true;}
  static void fatalMsg(const char *msg) { cerr << "ERROR: From Trax " << msg << endl; 		Globals::hasError = true;
}




}
;
#endif