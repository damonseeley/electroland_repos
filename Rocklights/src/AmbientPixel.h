/*
 *  OffsetPixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */
#ifndef __AMBIENTPIXEL_H__
#define __AMBIENTPIXEL_H__

#include "Panel.h"
#include "BasePixel.h"
#include "PersonStats.h"
#include "InterpGen.h"
#include <GLUT/glut.h>
#include "debug.h"

class PersonStats;
class BasePixel;

class Interpolator;

class AmbientPixel : public BasePixel {
  Interpolator *user; // the interpolator using the offsetpixel (will be notified if deleted)

  int r, g, b;

  float scale;

  BasePixel *pixel;


  

  // if we are at col, row and we add(r,b,c) then we add to col+colOffset r*curColOffset, g*curColOffset,.... 
  // same for row
  
public:
  void setUser(Interpolator* i) { user = i;}
  
  AmbientPixel(Panel *p, int col, int row, int stick = -1) ;
  ~AmbientPixel();

  // cOffset and rOffset are between -.5 and .5;

  void updateScale(float avatarScale); 


  void clear() { r = 0; g = 0; b= 0; }
    
  virtual void addColor(int r, int g, int b) ;
  virtual void addColor(int subPixel, int r, int g, int b) ;


  void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) { cr =r; cg = g; cb = b; }
  void getRGBData(int i, unsigned char &cr, unsigned char &cg,  unsigned char &cb) { cr =r; cg = g; cb = b; }
 
  virtual bool setScale(float s); // only used by ambientpixel

}
  ;

#endif