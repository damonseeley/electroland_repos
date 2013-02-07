/*
 *  Avatar .h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __AVATAR_H__
#define __AVATAR_H__

#include <string>
#include <vector>
#include "PersonStats.h"
#include "Panels.h"
#include "Interpolators.h"
#include "OffsetPixel.h"
#include "Arrangement.h"
//#include "IGeneric.h"

class OffsetPixel;
class Panels;
class PersonStats;
class Interpolators;
class Interpolator;
class Arrangement;

class Avatar  {
//  OffsetPixel *offsetPixel[MAXOFFSETPIXELSPERAVATAR];
//  int offsetPixelCnt;
	vector<OffsetPixel *> offsetPixelVec1;
	vector<OffsetPixel *> offsetPixelVec2;
	vector<OffsetPixel *> *curOffsetPixelVec;
	vector<OffsetPixel *> *oldOffsetPixelVec;



  float scale; // scale is used for crossfades (or fade outs) between avitars/patterns

private:
  int col;
  int row;
  int oldCol;
  int oldRow;
  int pil;
  int oldPil;

  int avColor;

  int tailDelay;
  int pillarMode;

  int renderPerdictorLength;


  static int pilInerp[3][7][25];
  static int tailFade[3][8] ;
  static int pillarLightSpeed;
  static int defTailDelay;

  static bool staticInit;

//  int redPredict[8];
//  int redFade[8];

  Interpolator *c0;
Interpolator *c1;
Interpolator *c2;
Interpolator *c3;
Interpolator *c4;
Interpolator *c5;
Interpolator *c6;

public:

  string name;
  Arrangement *enterArrangement;
  Arrangement *exitArrangement;
  Arrangement *moveArrangement;
  Arrangement *overheadArrangement;
  string enterSound;
  string exitSound;
  string moveSound;
  int enterSoundLoop;
  bool unEntered;

  Panels *panels;
  enum {A, B, C, D, E, F, G, H, I, J, MAXPANELS};

  enum {RED, GREEN, BLUE};

  



  Avatar () ;
  virtual ~Avatar ();
  
  // call update pixels here
  // then call new vitual function that does user specific updates...
  // un virutalize update
  void update(PersonStats *personStats, int ct, int dt, Interpolators *interps);

  void setName(string s) { name = s; }
  void setEnterArrangement(Arrangement *ar) { enterArrangement = ar; }
  void setExitArrangement(Arrangement *ar) { exitArrangement = ar; }
  void setMoveArrangement(Arrangement *ar) { moveArrangement = ar; }
  void setOverheadArrangement(Arrangement *ar) { overheadArrangement = ar; }

  void setEnterSound(string s) { enterSound = s; }
  void setExitSound(string s) { exitSound = s; }
  void setMoveSound(string s) { moveSound = s; }
  void setEnterSoundLoop(int i) { enterSoundLoop = i; }
  void setColor(int c) { avColor =c; }
  int getColor() { return avColor; }

  virtual void init(Interpolators *interps) {}; // do stuff with the above
  virtual void enter(Interpolators *interps) {};
  virtual void updateFrame(PersonStats *personStats, int ct, int dt, Interpolators *interps) {}
  virtual void move(int col, int row, Interpolators *interps) {}
  virtual void exit(Interpolators *interps) {}


  OffsetPixel *addOffsetPixel(int panelName, int c, int r, bool smooth = true);
  OffsetPixel *addOffsetPixel(int panelName, int c, int r, int stick, bool smooth = true);

  void removeOffsetPixel(OffsetPixel* o);

  void updateOffsetPixels(PersonStats *personStats, float dt);
  void clear();

//  virtual void destroy();

  float getScale() { return scale; }
  void setScale(float f) { scale = f; }
  void addColor(int panel, int col, int row, int r, int g, int b);
  void addColor(int panel, int col, int row, int stick, int r, int g, int b);


  void renderPerdictor(PersonStats *personStats, Interpolators *interps);


  void genericInterpDone(Interpolator *ig) {};
  
  void setTrailDelay(int usecs);
  void setPillarMode(int dir);
  void setRenderPerdictor(int i) { renderPerdictorLength = i; }
  void renderPillars(PersonStats *personStats, Interpolators *interps);
  void renderTail(PersonStats *personStats, Interpolators *interps);
  int getCol() {return col;}
  int getRow() { return row; }

} 
;
#endif