/*
 *  ColorElementR.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/25/05.
 * 
 *
 */

#include "ColorElementR.h"

ColorElementR::ColorElementR() : ColorElementRGB() {
  addType = MAX;
}


void ColorElementR::update() {
  if (! dirty) return;
  
  switch(addType) {
    case AVERAGE: {
      float aveScale = 1.0f / (float) counter;
      uR = (unsigned char) (r * aveScale);
                  }
      break;
    case MAX:
    case OVERWRITE:
    case CAP:
    case NORM: {
    default:
      r = (r < 0) ? 0 : r;
      uR = (r > 255) ? 255 : r;       
      break;
  }
  }
}
