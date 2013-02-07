/*
 *  Avatar .h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __AMBIENT_H__
#define __AMBIENT_H__

#include "WorldStats.h"
#include "Panels.h"
#include "Interpolators.h"
#include "MasterController.h"

class Panels;
class PersonStats;
class Interpolators;
class WorldStats;

class MasterController;

class Ambient  {
public:
	MasterController *caller;
	bool interpOwner;
  Interpolators *interps;
  float scale; // scale is used for crossfades (or fade outs) between avitars/patterns


  Panels *panels;
    BasePixel* pixels[STICKCNT];
    int pixelCnt;

  enum {A, B, C, D, E, F, G, H, I, J, MAXPANELS};

  Ambient(bool swapAmbients = true) ; // true same as no arg
  virtual ~Ambient();
  
  // call update pixels here
  // then call updateFrame that does user specific updates...
  // un virutalize update
  void update(WorldStats *worldStats, int ct, int dt, float curScale);

  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) {}

  AmbientPixel* Ambient::addAmbientPixel(int panelName, int c, int r, int stick = -1);

//  virtual void destroy();

  float getScale() { return scale; }
  void setScale(float f) { scale = f; }

  void addColor(int panel, int col, int row, int r, int g, int b);
  void addColor(int panel, int col, int row, int stick, int r, int g, int b);

  BasePixel *addPixel(int panelName, int c, int r);

  void setCaller(MasterController *mc) { caller = mc; }



} 
;

#endif