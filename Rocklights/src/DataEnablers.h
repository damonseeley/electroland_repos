/*
 *  DataEnablers.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#ifndef __DATAENABLERS_H__
#define __DATAENABLERS_H__

#include "KiNETLibSimple.h"
#include "DataEnablerFile.h"
#include "DataEnabler.h"
#include "globals.h"

class DataEnabler;
class DataEnablers {
public:
  int dataEnablerMaxCnt;
  DataEnabler *dataEnabler;

public:
    DataEnablers();
    ~DataEnablers() { delete ks; }
  static KiNETSimple *ks;
    void sendDMX();
    void setupEnablers(char* filename);
    void clear();

}
;
#endif