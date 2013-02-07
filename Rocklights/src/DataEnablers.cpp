/*
 *  DataEnablers.cpp
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/28/05.
 * 
 *
 */

#include "DataEnablers.h"

KiNETSimple *DataEnablers::ks = NULL;

DataEnablers::DataEnablers() {
  if (ks == NULL) {
  	ks = new KiNETSimple();
  }
}

void DataEnablers::sendDMX() {
  for (int i = 0; i < dataEnablerMaxCnt; i++) {
    dataEnabler[i].sendDMX();
  }
}

void DataEnablers::setupEnablers(char* filename) {
  DataEnablerFile *def = new DataEnablerFile(filename);
  def->read();
//  def->print();
  
  dataEnablerMaxCnt = def->getMaxId() + 1;
  
  dataEnabler = new DataEnabler[dataEnablerMaxCnt];

  
  for(int i = 0; i < dataEnablerMaxCnt; i++) {
    dataEnabler[i].set(def->ip[i], def->size[i]);
  }

}
void DataEnablers::clear() {

    for(int i = 0; i < dataEnablerMaxCnt; i++) {
    dataEnabler[i].clear();
  }

}


