#include "InterpGen.h"

InterpGen::InterpGen() {
    isRunning = false;
    wasStarted = false;
}

  void InterpGen::start(int timeUSecs) {
    deltaPerUSec = 1.0f / ((float) timeUSecs);
    perc = 0.0;
    isRunning = true;
    wasStarted = true;
  }

  
  void InterpGen::reset() {
    wasStarted = false;
  }

  float InterpGen::update(int deltaT) {
    if(isRunning) {
      perc += (deltaPerUSec * deltaT);
      if (perc >= 1.0f) {
        perc = 1.0f;
        isRunning = false;
      }
      
    }

    return perc;

  }

