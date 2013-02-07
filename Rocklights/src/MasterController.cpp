#include "MasterController.h"

Ambient *MasterController::curAmbient = NULL;
Ambient *MasterController::oldAmbient = NULL;

int MasterController::transitionCrossFadeTime = -1;
int MasterController::globalTransitionCrossFadeTime = -1;
MasterController *MasterController::curMasterController = NULL;
InterpGen *MasterController::interpGen = NULL;
bool MasterController::isInTransition = false;
bool MasterController::localTrans = true;


MasterController::MasterController(PeopleStats *peopleStats) {
	name = "";
  if (curMasterController == NULL) { // first time
    curMasterController = this;
    transitionCrossFadeTime = CProfile::theProfile->Int("transitionCrossFadeTime", 1000);
	globalTransitionCrossFadeTime = transitionCrossFadeTime;
    interpGen = new InterpGen();
    curAmbient = new Ambient();
    oldAmbient = new Ambient();
    
  } else {
    delete curMasterController;
    curMasterController = this;
    delete oldAmbient;
    oldAmbient = curAmbient;
  }
  
//  interpGen->start(transitionCrossFadeTime);
 
  localTrans = true;
}

MasterController::~MasterController() {
}


void MasterController::update(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {

  isInTransition = localTrans || ps->isInTransition;
 



  if (localTrans) {
    float scale;
    if(interpGen->wasStarted) {
      if(interpGen->isRunning) {
        scale = interpGen->update(dt);
      } else {
        interpGen->reset();
        localTrans = false;
        isInTransition = false;
        scale = 1.0f;
      }
	} else {
      localTrans = true;
      interpGen->start(transitionCrossFadeTime);
      scale = 0.0f;
    }
    if (curAmbient) curAmbient->update(worldStats, ct, dt, scale);
    if (oldAmbient) oldAmbient->update(worldStats, ct, dt, 1.0f - scale);
  } else {
    if (curAmbient) curAmbient->update(worldStats, ct, dt, 1.0f);
  }
  updateFrame(worldStats, ps, ct, dt);

}  
