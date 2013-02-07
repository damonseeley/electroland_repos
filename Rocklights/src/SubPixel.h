/*
 *  SubPixel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */
#ifndef __SUBPIXEL_H__
#define __SUBPIXEL_H__

#include "BasePixel.h"
#include "LightElement.h"
#include <GLUT/glut.h>

class SubPixel : public BasePixel {
  
  LightElement *le;

  int addCount;
  bool dirty;
  

  
public:
  int r;
  int g;
  int b;
  

  
  
  
  SubPixel(LightElement *el);
  
  void clear();
  void update(); // updates dmx data
  void display(); // writes to openGL

  void addColor(int r, int g, int b);


  // get the RGB data from the data enabler
  void getRGBData(unsigned char &cr, unsigned char &cg,  unsigned char &cb) { le->getRGBData(cr, cg, cb); }
  
  void setDims(float t, float l, float b, float r);
}
;

#endif