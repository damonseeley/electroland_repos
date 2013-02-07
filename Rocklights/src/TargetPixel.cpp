/*
 *  TargetPixel.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#include "TargetPixel.h"


TargetPixel::TargetPixel(LightElement *st0) : BasePixel() {
  target = new SubPixel(st0);
  if (st0->lightType != LightElement::TARGET) {
    timeStamp(); std::clog << "Attempt to use non-taget LightElement with TargetPixel\n";
  }
  rotA = trotA = 0.0f;
  x = y = z = -1;
  isTarget = true;

}


void TargetPixel::clear() {
  target->clear();
}

void TargetPixel::addColor(int r, int g, int b) {
  target->addColor(r,g,b);
}

void TargetPixel::addColor(int subPixel, int r, int g, int b) {
  target->addColor(r,g,b);
}


void TargetPixel::update() {
  target->update();
}

void TargetPixel::display() {
  //  std::co ut << "Drawlight at " << x << " " << y << " " << z << std::endl;
  glPushMatrix();
  glTranslatef(x,y,z);	
  
  if(rotA != 0.0f) {
    glRotatef(rotA, rotX, rotY, rotZ);
  }
  drawQuad();
  
  glPopMatrix();
}

void TargetPixel::topDisplay() {
  //  std::co ut << "Drawlight at " << x << " " << y << " " << z << std::endl;
  glPushMatrix();
  glTranslatef(tx,ty,tz);	
  
    if(trotA != 0.0f) {
      glRotatef(trotA, trotX, trotY, trotZ);
    }
  drawQuad();
  
  glPopMatrix();
}

void TargetPixel::drawQuad() {  
  target->display();
}


void TargetPixel::setDims(float t, float l, float b, float r) { 
  top =t; left= l; bot = b; right = r; 
    
  width = right - left;
  height = bot - top;
  halfWidth = width * 0.5f;
  halfHeight = height * 0.5f;

  target->setDims(t, l, b, r);
  
}


