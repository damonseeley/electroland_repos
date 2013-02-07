/*
 *  Pixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */


#ifndef __BASEPIXEL_H__
#define __BASEPIXEL_H__


#include "LightElement.h"
#include <GLUT/glut.h>
//#include "Interpolator.h"

#include <iostream>


class BasePixel {


public:
  static BasePixel *blank;

  float x, y, z;
  float tx, ty, tz;
  float rotA, rotX, rotY, rotZ;
  float trotA, trotX, trotY, trotZ;
  
  
  float top, left, bot, right;
  float width;
  float height;
  float halfWidth;
  float halfHeight;

  bool isTarget;


  int addMode;
  

  BasePixel();

  virtual ~BasePixel(){}

  static void initBlank() { blank = new BasePixel(); }

  enum AddModeTypes { CAP, MAX, NORM, OVERWRITE, AVERAGE };
  enum PixelTypes { BASE, AMBIENT, OFFSET, SUBPIXEL } ;
  int pixelType;
  
  inline int getMax(int a, int b, int c) { return (a > b) ? ((a > c) ? a : c)  : ((b > c) ? b : c); }

  
  
  virtual void clear() {}
  
  virtual void addColor(int r, int g, int b) {}
  virtual void addColor(int subPixel, int r, int g, int b) {}

  virtual void update() {} // updates dmx data
  virtual void display() {} // writes to openGL
  virtual void topDisplay() {} // writes to openGL
  
  void setMode(int mode) { addMode = mode; }
  
  // get the RGB data from the data enabler
  virtual void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) {  }
  virtual void getRGBData(int subPixel, unsigned char &cr, unsigned char &cg,  unsigned char &cb) {  }

  
  virtual void setDims(float t, float l, float b, float r) {}

  virtual BasePixel* getSubPixel(int i) { return blank; }

  void setPos(float fx, float fy, float fz) { x = fx, y = fy, z = fz;}
  void setTopPos(float fx, float fy, float fz) { tx = fx, ty = fy, tz = fz; }
  void setRot(float fa, float fx, float fy, float fz) { rotA = fa; rotX = fx; rotY = fy; rotZ =fz; }
  void setTopRot(float fa, float fx, float fy, float fz) { trotA = fa; trotX = fx; trotY = fy; trotZ =fz; }


  bool virtual setScale(float s) { return false; }; // only used by ambientpixel

}
;



#endif