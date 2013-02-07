/*
 *  LightFileDimGather.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#include "LightFile.h"


LightFile::LightFile(const char *dmxFilename) {
  minCol = new int[Panels::PANEL_CNT];
  maxCol = new int[Panels::PANEL_CNT];
  minRow = new int[Panels::PANEL_CNT];
  maxRow = new int[Panels::PANEL_CNT];
  
  filenameStr = dmxFilename;
//  locFilenameStr = locFileName;


  if(open(file, dmxFilename)) {
    timeStamp(); clog << "INFO  Opened light file: " << dmxFilename << "\n";
  } else {
    timeStamp(); clog << "ERROR  Unable to open file: " << dmxFilename << endl;
			Globals::hasError = true;

  }
  
  /*
  if(open(locFile, locFileName)) {
    co ut << "Opened Light Location file: " << locFileName << endl;
  } else {
    cou t << "ERROR: unable to open file: " << locFileName << endl;
  }
  */
  
  for(int i = 0; i < Panels::PANEL_CNT; i++) {
    minCol[i] = INT_MAX;
    minRow[i] = INT_MAX;
    maxCol[i] = -INT_MAX;
    maxRow[i] = -INT_MAX;
  }
  line = 0;
  scale = 1;
  
}

int LightFile::getWidth(int panel) { 
  if (minCol[panel] == INT_MAX) {
    return -1;
  }
  return 1 + maxCol[panel] - minCol[panel]; 
}
int LightFile::getHeight(int panel) { 
  if (minRow[panel] == INT_MAX) {
    return -1;
  }
  return 1 + maxRow[panel] - minRow[panel]; 
}

LightFile::~LightFile() {
  close();
}


bool LightFile::open(ifstream &file, const char *filename) {
  file.open(filename, ios::in);
  return file.is_open();
}


bool LightFile::readLine(char &let, int &col, int &row, float &x, float &y, float &z, int &dataEnablerID, int &channel) {
  // temporary strings for reading lines
  string letStr;
  string colStr;
  string rowStr;
  string xStr;
  string yStr;
  string zStr;
  string dataStr;
  string chanStr;
  
  
  if (getline(file, letStr, ',').eof()) return false;
  if (getline(file, colStr, ',').eof()) return false;
  if (getline(file, rowStr, ',').eof()) return false;
  if (getline(file, xStr, ',').eof()) return false;
  if (getline(file, yStr, ',').eof()) return false;
  if (getline(file, zStr, ',').eof()) return false;
  if (getline(file, dataStr, ',').eof()) return false;
  if (getline(file, chanStr).eof()) return false;
  
  let = letStr[0];
  if((let < 'A') || (let > 'J')) {
    timeStamp(); clog << "ERROR  Malformed Panel Letter (Letter must be between A and J)" << endl;
		Globals::hasError = true;

    return false;
  }
  
  col = atoi(colStr.c_str());
  row = atoi(rowStr.c_str());
  x = atof(xStr.c_str()) * scale;
  y = atof(yStr.c_str()) * scale;
  z = atof(zStr.c_str()) * scale;
  dataEnablerID = atoi(dataStr.c_str());
  channel = atoi(chanStr.c_str()); 
  
  line++;
  
  
  return true;
  
}

void LightFile::readWidthHeight() {
  
  char c;
  int col;
  int row;
  float x;
  float y;
  float z;
  int de;
  int ch;
  
  while(readLine(c, col, row, x, y, z, de, ch)) {
    int curC = c - 'A';
    minCol[curC] = (minCol[curC] < col) ? minCol[curC] : col;
    minRow[curC] = (minRow[curC] < row) ? minRow[curC] : row;
    maxCol[curC] = (maxCol[curC] > col) ? maxCol[curC] : col;
    maxRow[curC] = (maxRow[curC] > row) ? maxRow[curC] : row;
//    cou t << "line " << line <<":  " << c << "  " << col <<", " << row << "  " << x << ", " << y << ", " << z << endl;
  }
  reopen();
}

void LightFile::reopen() {
  close();
  file.clear();
  open(file, filenameStr.c_str());
}

  

