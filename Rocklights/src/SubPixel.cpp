/*
 *  SubPixel.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#include "SubPixel.h"

SubPixel::SubPixel(LightElement *el) : BasePixel() {
  le = el;
  addMode = CAP;
  pixelType = SUBPIXEL;
  clear();
}


void SubPixel::clear() {
  r = g = b = addCount = 0;
  dirty = false;
}

void SubPixel::addColor(int cr,  int cg, int cb) {
  dirty = true;
  switch(addMode) {
    case AVERAGE: // no break on purpose.  Add average like others
      addCount++;
    case CAP:
    case NORM:
      r += cr; g += cg; b+=cb;

      break;
    case MAX:
      r = (cr > r) ? cr : r;
      g = (cg > g) ? cg : g;
      b = (cb > b) ? cb : b;      
      
      break;
    case OVERWRITE:
      r = cr; g = cg; b = cb;
      break;
    default:
      r += cr; g += cg; b+=cb;
      break;      
  }
}

void SubPixel::update() {
  if (! dirty) return;
  
  switch(addMode) {
    case CAP:
    case MAX:
    case OVERWRITE:
      break;
    case NORM: {
      int maxC = getMax(r,g,b);
      
      if (maxC > 255) {
        float normScale = 255.0f / (float) maxC;
        r = ((float) r * normScale);
        g = ((float) g * normScale);
        b = ((float) b * normScale);
      }              
               }
      break;
    case AVERAGE: {
      float aveScale = 1.0f / (float) addCount;
      r = ((float) r * aveScale);
      g = ((float) g * aveScale);
      b = ((float) b * aveScale);
                  }
      break;
                  
    default:
      break;
  }

      r = (r > 255) ? 255 : r;
      g = (g > 255) ? 255 : g;
      b = (b > 255) ? 255 : b;
      r = (r < 0) ? 0 : r;
      g = (g < 0) ? 0 : g;
      b = (b < 0) ? 0 : b;      

      le->setColor(r, g, b);
  clear();
}

void SubPixel::display() {
  float fR, fG, fB;
  unsigned char uR, uG, uB;
  le->getRGBData(uR, uG, uB);

  fR = (float) uR / 255.0f;
  fG = (float) uG / 255.0f;
  fB = (float) uB / 255.0f;

  
  if((fR == 0) && ((fG == 0) && (fB == 0))) {
    glBegin(GL_LINE_LOOP);
    glColor3f(.7f, .7f, .7f);		
//    glColor3f(.0f, .0f, .0f);		
  } else {
    glBegin(GL_QUADS);
    glColor3f(fR, fG, fB);		
  }
  glVertex3f(left, top,0.0f);		
  glVertex3f(right, top,0.0f);			
  glVertex3f(right, bot,0.0f);			
  glVertex3f(left,  bot,0.0f);		
  glEnd();					
  }

void SubPixel::setDims(float t, float l, float b, float r) { 
  top =t; left= l; bot = b; right = r; 

  width = right - left;
  height = bot - top;
  halfWidth = width * 0.5f;
  halfHeight = height * 0.5f;

  /*
  std::co ut << "   top: " << t;
  std::co ut << "   bot: " << b;
  std::co ut << "   lef: " << l;
  std::co ut << "   rtg: " << r << std::endl;
  */
}

