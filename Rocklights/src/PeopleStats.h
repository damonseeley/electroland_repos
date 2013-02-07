/*
 *  PeopleStats.h
 *  RockefellerCenter
 *
 *  Created by Eitan Mendelowitz on 9/30/05.
 * 
 *
 */

#ifndef __PEOPLESTATS_H__
#define __PEOPLESTATS_H__

#include "PersonStats.h"
#include "Pattern.h"
#include "InterpGen.h"
#include <iostream>
using namespace std;

class Pattern;
class PersonStats;

class PeopleStats {
  int size;
  PersonStats **stats;
  // total ordering of everying
  PersonStats *headTot;


  int peopleCnt;

  float renderSize;


  int oldAvatarGroup;

  enum TRANS { NOTRANS, TOBLACK, TOOTHER };
  int transType;

  InterpGen *interpGen;



public:
  static PeopleStats *thePeopleStats;

  bool isInTransition;

  static int curAvatarGroup;

  PeopleStats(int hashSize);
  ~PeopleStats();

  void add(PersonStats *ps);
  bool remove(PersonStats *ps);
  bool remove(unsigned long i);
  bool removeAndDestroy(PersonStats *ps);
  bool removeAndDestroy(unsigned long i);


  int getSize() { return size; }


  PersonStats *get(unsigned long id);
  PersonStats *getHead();

  
PersonStats *PeopleStats::getSE();

PersonStats *PeopleStats::getRandom();
  
  void display();

  void update(int curTime, int deltaTime);

  void transitionTo(Pattern *newPattern, int uSecs);




}
;

#endif