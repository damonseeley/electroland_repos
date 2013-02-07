/*
 *  ColorElementRGB.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/24/05.
 * 
 *
 */

#include "ColorElementRGB.h"

//enum addMethodTypes { CAP, MAX, NORM, OVERWRITE, AVERAGE };

ColorElementRGB::ColorElementRGB() {
  clear();
  addType = CAP;
}






void ColorElementRGB::addColor(unsigned char cr, unsigned char cg, unsigned char cb) {
  dirty = true;
  switch(addType) {
    case AVERAGE: // no break on purpose.  Add average like others
      counter++;
    case CAP:
    case NORM:
      r += cr; g += cg; b+=cb;
      break;
    case MAX:
      setMaxR(cr); setMaxG(cg); setMaxB(cb);
      break;
    case OVERWRITE:
      r = cr; g = cg; b = cb;
      break;
    default:
      r += cr; g += cg; b+=cb;
      break;      
      
  }
}



void ColorElementRGB::update() {
  if (! dirty) return;
  
  switch(addType) {
    case CAP:
      uR = (r > 255) ? 255 : r; 
      uG = (g > 255) ? 255 : g; 
      uB = (b > 255) ? 255 : b; 
      break;
    case MAX:
    case OVERWRITE:
      uR = r;
      uG = g;
      uB = b; // no break on purpose
    case NORM: {
      int sum = r + g + b;
      float normScale = 255.0f / (float) sum;
      uR = (unsigned char) (r * normScale);
      uG = (unsigned char) (g * normScale);
      uB = (unsigned char) (b * normScale); }
      break;
    case AVERAGE: {
      float aveScale = 1.0f / (float) counter;
      uR = (unsigned char) (r * aveScale);
      uG = (unsigned char) (g * aveScale);
      uB = (unsigned char) (b * aveScale); }
      break;
    default:
      uR = (r > 255) ? 255 : r; 
      uG = (g > 255) ? 255 : g; 
      uB = (b > 255) ? 255 : b; 
      break;
  }
}
