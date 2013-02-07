/*
 *  Panel.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#ifndef __PANEL_H__
#define __PANEL_H__

#include "BasePixel.h"
#include "debug.h"
#include "float.h"

#include <iostream>
using namespace std;
class BasePixel;

class Panel {

  bool isPillar; 
   int width;
  int height;
  int size;


public:

  float minX, minY, minZ;
  float maxX, maxY, maxZ;

  BasePixel** pixels;
  int *xCalc;
  int *yCalc;

  // pre compute distances between ajacent pixels
  
  float *distEast;
  float *distSouth;
  float *distSouthEast;

  Panel();  
  Panel(int w, int h);
  ~Panel();
  void set(int w, int h);
  
  void setPixel(int col, int row, BasePixel *pixel);
  BasePixel *getPixel(int col, int row);
  BasePixel *getPixel(int row); // only should be used on signel column panels
  BasePixel *getPixelRow(int row); // give a whole row.  Should be faster for block access
  int getWidth() { return width; }
  int getHeight() { return height; }
  
  void display();
  void topDisplay();
  void update();

  void calcStats();
  void calcARow();
  void calcACol();

  void calcDistGrid();
  //void calcZ();
  
  int getRow(float y);
  int getCol(float x);
//  int getElevation(float z);

  // results for A will be pretty meaningless
  // row shouldn't make a difference for anything but B
  float floorDistSquared(float x, float y, float row);
}
;
#endif