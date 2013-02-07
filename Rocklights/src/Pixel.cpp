/*
 *  Pixel.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/27/05.
 * 
 *
 */

#include "Pixel.h"

Pixel::Pixel(LightElement *st0, LightElement *st1, LightElement *st2, LightElement *st3) : BasePixel() {
  if (st0->lightType != LightElement::RGB) {
    timeStamp(); std::clog << "WARNING  Attempt to use non-RGB LightElement with Pixel" << "/n";
  }
  if (st1->lightType != LightElement::RGB) {
    timeStamp(); std::clog << "WARNING  Attempt to use non-RGB LightElement with Pixel" << "/n";
  }
  if (st2->lightType != LightElement::RGB) {
    timeStamp(); std::clog << "WARNING  Attempt to use non-RGB LightElement with Pixel" << "/n";
  }
  if (st3->lightType != LightElement::RGB) {
    timeStamp(); std::clog << "WARNING  Attempt to use non-RGB LightElement with Pixel" << "/n";
  }
  
  stick[0] = new SubPixel(st0);
  stick[1] = new SubPixel(st1);
  stick[2] = new SubPixel(st2);
  stick[3] = new SubPixel(st3);
  rotA = trotA = 0.0f;
  x = y = z = -1;
}


void Pixel::clear() {
  stick[0]->clear();
  stick[1]->clear();
  stick[2]->clear();
  stick[3]->clear();
}


void Pixel::addColor(int r, int g, int b) {
  stick[0]->addColor(r,g,b);
  stick[1]->addColor(r,g,b);
  stick[2]->addColor(r,g,b);
  stick[3]->addColor(r,g,b);
}

void Pixel::addColor(int subPixel, int r, int g, int b) {
  stick[subPixel]->addColor(r,g,b);
}

BasePixel* Pixel::getSubPixel(int i) { 
  
  ASSERT((i >= 0) && (i < 4), "Subpixel must be between 0 and 1");
  
  return stick[i]; 
  
}

void Pixel::update() {
  stick[0]->update();
  stick[1]->update();
  stick[2]->update();
  stick[3]->update();
}
void Pixel::display() {
//  std::cou t << "Drawlight at " << x << " " << y << " " << z << std::endl;
  glPushMatrix();
  glTranslatef(x,y,z);	
  
    if(rotA != 0.0f) {
     glRotatef(rotA, rotX, rotY, rotZ);
   }
  drawQuad();
  
  glPopMatrix();
}

void Pixel::topDisplay() {
  //  std::c out << "Drawlight at " << x << " " << y << " " << z << std::endl;
  glPushMatrix();
  glTranslatef(tx,ty,tz);	
  
  if(trotA != 0.0f) {
    glRotatef(trotA, trotX, trotY, trotZ);
  }
  drawQuad();
  
  glPopMatrix();
}

void Pixel::drawQuad() {
  
  stick[0]->display();
  stick[1]->display();
  stick[2]->display();
  stick[3]->display();
}


void Pixel::setDims(float t, float l, float b, float r) { 
  top =t; left= l; bot = b; right = r; 
  
  
  width = right - left;
  height = bot - top;
  halfWidth = width * 0.5f;
  halfHeight = height * 0.5f;

  float qw = (r - l) * .25;
  
  stick[0]->setDims(t, l, b, l + qw);
  stick[1]->setDims(t, l + qw, b, l + (2 * qw));
  stick[2]->setDims(t, l + (2 * qw) , b, l + (3 * qw));
  stick[3]->setDims(t, l + (3 * qw), b, r);
  
/*
  float qh = (b - t) * .25;
  
  stick[0]->setDims(t, l, t + qh, r);
  stick[1]->setDims(t + qh, l, t + (2 * qh), r);
  stick[2]->setDims(t + (2 * qh), l, t + (3 * qh), r);
  stick[3]->setDims(t + (3 * qh), l, b, r);
*/  
}


