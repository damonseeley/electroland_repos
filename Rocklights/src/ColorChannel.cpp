/*
 *  ColorChannel.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#include "ColorChannel.h"

void ColorChannel::display() {
  if(color == 0) {
    glBegin(GL_LINE_LOOP);
    glColor3f(.3f, .3f, .3f);		
  } else {
    glBegin(GL_QUADS);
    float fc = ((float) *color) * (1.0f / 255.0f);
    glColor3f(fc, fc, fc);
  }

  glVertex3f(left, top, 0.0f);		
  glVertex3f(right, top, 0.0f);			
  glVertex3f(right, bot,0.0f);			
  glVertex3f(left, bot,0.0f);		
  glEnd();					


}