/*
 *  TargetPixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#ifndef __TARGETPIXEL_H__
#define __TARGETPIXEL_H__

#include "BasePixel.h"
#include "LETargetCircle.h"
#include "SubPixel.h"
#include "Globals.h"
#include <GLUT/glut.h>

class BasePixel;
class TargetPixel : public BasePixel {
  
  SubPixel *target;
  
  
public:
  
  
  
  
  TargetPixel(LightElement *st0);
  
  void clear();
  void update(); // updates dmx data
  void display(); // writes to openGL
  void topDisplay();
  
  void setMode(int mode) { addMode = mode; }
  
  virtual void addColor(int r, int g, int b);
  virtual void addColor(int subPixel, int r,int g, int b) ;

  BasePixel* getSubPixel(int i) { return target; }
  
  // get the RGB data from the data enabler
  void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) { target->getRGBData(cr, cg, cb); }
  void getRGBData(int i, unsigned char &cr, unsigned char &cg,  unsigned char &cb) { target->getRGBData(cr, cg, cb); }
  
  void setDims(float t, float l, float b, float r);
  void drawQuad();
}
;

#endif