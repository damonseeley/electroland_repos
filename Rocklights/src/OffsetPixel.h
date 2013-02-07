/*
 *  OffsetPixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */
#ifndef __OFFSETPIXEL_H__
#define __OFFSETPIXEL_H__

#include "Panel.h"
#include "BasePixel.h"
#include "PersonStats.h"
#include "InterpGen.h"
#include <GLUT/glut.h>
#include "debug.h"

class PersonStats;
class BasePixel;

class Interpolator;

class OffsetPixel : public BasePixel {
    Interpolator *user; // the interpolator using the offsetpixel (will be notified if deleted)


  static int crossFadeTime;

  bool isSmooth;
  int r, g, b;

  InterpGen *interp;

  int colOffset;
  int rowOffset;

  float scale;

  float scaleInv;

 
  Panel *panel;


  float cutOffPercent;

  BasePixel *oldPixel;
  BasePixel *newPixel;

 // int id;     //  UNDONE remove
//  static idCnt; // UNDONE remove

  int stick;
public:
	  bool needsReaping;

  

  // if we are at col, row and we add(r,b,c) then we add to col+colOffset r*curColOffset, g*curColOffset,.... 
  // same for row
  
public:
  void setUser(Interpolator* i) { user = i;}
  
  OffsetPixel(Panel *p, int cOffset, int rOffset, bool smooth = true) ;
  OffsetPixel(Panel *p, int cOffset, int rOffset, int stick, bool smooth = true) ;
  ~OffsetPixel();

  // cOffset and rOffset are between -.5 and .5;

  void updatePosition(PersonStats *personStats, int deltaT, float avatarScale); 


  void clear() { r = 0; g = 0; b= 0; }
    
  virtual void addColor(int r, int g, int b);
  virtual void addColor(int subPixel, int r, int g, int b);


  // get the RGB data from the data enabler
  void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) { cr =r; cg = g; cb = b; }
  void getRGBData(int i, unsigned char &cr, unsigned char &cg,  unsigned char &cb) { cr =r; cg = g; cb = b; }
  
  void setDims(float t, float l, float b, float r){}

  void println() { cout << "(" << colOffset << ", " << rowOffset << ")-" << needsReaping << endl; }
}
  ;

#endif