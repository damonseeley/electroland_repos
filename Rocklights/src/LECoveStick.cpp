/*
 *  LECoveStick.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#include "LECoveStick.h"


LECoveStick::LECoveStick(unsigned char *data, int offSet) {
  r = &data[offSet];
  g = &data[offSet + 1];
  b = &data[offSet + 2];
  lightType = RGB;
}

void LECoveStick::setDataChannel(unsigned char *data, int offSet) {
  r = &data[offSet];
  g = &data[offSet + 1];
  b = &data[offSet + 2];
}

void LECoveStick::setColor(unsigned char rVal, unsigned char gVal, unsigned char bVal) {
  *r = rVal;
  *g = gVal;
  *b = bVal;
}

void LECoveStick::getRGBData(unsigned char &cr, unsigned char &cg, unsigned char &cb) {
  cr = *r;
  cg = *g;
  cb = *b;
}


