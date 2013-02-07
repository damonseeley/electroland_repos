/*
 *  Panels.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */
#ifndef __PANELS_H__
#define __PANELS_H__


#include <iostream>
using namespace std;

#include "globals.h"
#include "LightFile.h"
#include "DataEnabler.h"
#include "LECoveStick.h"
#include "Pixel.h"
#include "LETargetCircle.h"
#include "Targetpixel.h"
#include "Panel.h"

class LightFile;

class Panel;

class Panels {
  float halfWidth; 
  float halfHeight;
  float targetRadius;
  

  
public:
  enum { A, B, C, D, E, F, G, H, I, J, PANEL_CNT };
  Panel panels[PANEL_CNT];
  
  Panels(float w, float h, float r);
  
  void setLights( LightFile *lightFile, DataEnabler dataEnablers[], bool isTarget);
  
  void setPixel(int let, int col, int row, BasePixel *pixel);
  
  BasePixel* getPixel(int let, int col, int row);
  BasePixel* getPixel(int let, int col);
  
  void print();
  int getWidth(int i) { return panels[i].getWidth() ; }
  int getHeight(int i) { return panels[i].getHeight(); }
  
  void update();
  void display();
  void topDisplay();

  
  static Panels *thePanels;
  
}
;
#endif