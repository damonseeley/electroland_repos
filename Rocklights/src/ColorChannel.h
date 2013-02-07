/*
 *  ColorChannel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#ifndef __COLORCHANNEL_H__
#define __COLORCHANNEL_H__

#include <GLUT/glut.h>


class ColorChannel {
private:
  unsigned char* color;
  float top, bot, left, right;
  
public:
  ColorChannel(unsigned char* c) { color = c; }  
  // sets color in data array that is used for DMX sending
  void setColor(unsigned char c) {  *color = c; }
  void setDims(float t, float l, float b, float r) { top = t; left = l; bot = b; right = r; }
  
  void display() ;

}
;

#endif