/*
 *  LightFile.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */


#ifndef __LIGHTFILE_H__
#define __LIGHTFILE_H__

#include <iostream>
#include<iomanip>
#include<fstream>
#include <string>
using namespace std;

#include "DataEnabler.h"
#include "Panels.h"

class Panels;

class LightFile {
  string filenameStr;
//  string locFilenameStr;

  ifstream file;
  
  int line;
  float scale; // scaler to cm
private:
  int *minCol;
  int *minRow;
  int *maxCol;
  int *maxRow;

  float rowCenters[50];
  float colCenters[50];
public:

  

    LightFile(const char *dmxFileName);
  ~LightFile();
  bool open(ifstream &file, const char *filename);
  void close() { file.close();  }
  void reopen();
  bool readLine(char &let, int &col, int &row, float &x, float &y, float &z, int &dataEnablerID, int &channel);
  void readWidthHeight();
  int getWidth(int panel);   
  int getHeight(int panel);
  int getMinRow(int i) { return minRow[i]; }
  int getMinCol(int i) { return minCol[i]; }
  int getMaxRow(int i) { return maxRow[i]; }
  int getMaxCol(int i) { return maxCol[i]; }
  void setMinRow(int i, int v) { minRow[i] = v; }
  void setMinCol(int i, int v) { minCol[i] = v; }
  void setMaxRow(int i, int v) { maxRow[i] = v; }
  void setMaxCol(int i, int v) { maxCol[i] = v; }
  void setScale(float f) { scale = f; }
  
}
;

#endif