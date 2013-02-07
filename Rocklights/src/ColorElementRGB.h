/*
 *  ColorElementRGB.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/24/05.
 * 
 *
 */


#ifndef _COLORELEMENTRGB_H_
#define _COLORELEMENTRGB_H_

#include "ColorElement.h"
#include<fstream>

class  ColorElementRGB : public ColorElement {
  
protected:
  int r;
  int g;
  int b;
  
  int counter;
  bool dirty;
  
  inline void setMaxR(unsigned char c) { r = (c > r) ? c : r; } 
  inline void setMaxG(unsigned char c) { g = (c > g) ? c : g; }
  inline void setMaxB(unsigned char c) { b = (c > b) ? c : b; }
public:
  
  ColorElementRGB();
  virtual ~ColorElementRGB() {}
  void addColor(unsigned char cr, unsigned char cg, unsigned char cb);

  void setColor(unsigned char cr, unsigned char cg, unsigned char cb) { r = cr; g = cg; b = cb; dirty = true; }
  void setR(unsigned char c) { r = c; dirty = true;}
  void setG(unsigned char c) { g = c; dirty = true;}
  void setB(unsigned char c) { b = c; dirty = true;}
  
  void getIntColor(int &cr, int &cg, int &cb) { cr =r; cg = g; cb = b; }
  inline int getIntR() { return r; }
  inline int getIntG() { return g; }
  inline int getIntB() { return b; }

  inline void clear(){ r=0; g=0; b=0; uR = 0; uG = 0; uB =0; counter=0; dirty = false; }

  virtual void update();
  virtual void render() {}
}
;

//  enum addMethodTypes { CAP, MAX, NORM, AND, OR, XOR, OVERWRITE, AVERAGE };




#endif