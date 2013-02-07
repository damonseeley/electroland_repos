
#ifndef __MASTERCONTROLLER__
#define __MASTERCONTROLLER__

#include "WorldStats.h"
#include "PeopleStats.h"
#include "InterpGen.h"
#include "Ambient.h"
#include <sstream>
class Ambient;
class MasterController {
public:
	string name;
	stringstream composerId;
  static MasterController *curMasterController;
  static Ambient* curAmbient;
  static Ambient* oldAmbient;
  static InterpGen *interpGen;
  static int globalTransitionCrossFadeTime;
  static int transitionCrossFadeTime;
  static bool isInTransition;
  static bool localTrans;
  int timeToIntermission; // noly used by MCGeneric



  MasterController(PeopleStats *peopleStats);
  virtual ~MasterController();

  void update(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

  virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {}


}

;
#endif