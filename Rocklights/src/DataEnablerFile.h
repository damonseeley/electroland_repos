/*
 *  DataEnablerFile.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#include <iostream>
#include<iomanip>
#include<fstream>
#include <string>
#include "globals.h"

using namespace std;

#ifndef __DATAENABLERFILE_H__
#define __DATAENABLERFILE_H__

class DataEnablerFile {

  ifstream file;
  string filenameStr;
  
public:
    enum { MAXENABLERS = 50 };  
  
  char ip[MAXENABLERS][20];
  int size[MAXENABLERS];

  int maxid;
  int line;

  DataEnablerFile(const char *fileName);
  ~DataEnablerFile();
  bool open(const char *fileName);
  void close() { file.close(); }
  void reopen();
  bool readLine(int &id, string &ip, int &size);
  int getMaxId() { return maxid; }
  void read();
  void print();

  
  bool isWhitespace(const char& c);
  string trim(string x);
  
  
}
;




#endif