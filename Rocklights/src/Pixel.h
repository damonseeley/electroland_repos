/*
 *  Pixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */
#ifndef __PIXEL_H__
#define __PIXEL_H__

#include "BasePixel.h"
#include "LECoveStick.h"
#include "SubPixel.h"
#include <GLUT/glut.h>
#include "debug.h"
#include "globals.h"

class BasePixel;

class Pixel : public BasePixel {
  
  SubPixel *stick[4];
private:
  void drawQuad();
  
public:

  
  
  
  Pixel(LightElement *st0, LightElement *st1, LightElement *st2, LightElement *st3);
  
  void clear();
  void update(); // updates dmx data
  virtual void display(); // writes to openGL
  virtual void topDisplay(); // writes to openGL
  
  BasePixel* getSubPixel(int i);
  
  
  void setMode(int mode) { addMode = mode; }
  
  virtual void addColor(int  r,  int g,int b);
  virtual void addColor(int subPixel, int r, int g, int b) ;



  // get the RGB data from the data enabler
  void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) { stick[0]->getRGBData(cr, cg, cb); }
  void getRGBData(int i, unsigned char &cr, unsigned char &cg,  unsigned char &cb) { stick[i]->getRGBData(cr, cg, cb); }
  
  void setDims(float t, float l, float b, float r);
}
  ;

#endif