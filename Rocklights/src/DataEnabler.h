/*
 *  DataEnabler.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/26/05.
 * 
 *
 */

#ifndef __DATAENABLER_H__
#define __DATAENABLER_H__
#include <iostream>
#include <stdlib.h>
#include "DataEnablers.h"
#include "globals.h"

#include "KiNETLibSimple.h"
class DataEnabler {
public:
  unsigned char *data;
  char* address;
  int size;
  
public:
    DataEnabler();
  DataEnabler(char* ip, int dataSize);
  ~DataEnabler();

  void set(char* ip, int dataSize);
  
  void clear();
  void print();
  void println();
  void printNonZero();
  void sendDMX();
 
}

;

#endif
