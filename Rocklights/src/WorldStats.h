#ifndef __WORLDSTATS_H__
#define __WORLDSTATS_H__

#include "InterpGen.h"
#include "PersonStats.h"
#include <iostream>
using namespace std;

class PersonStats;

class WorldStats {
public:


  int peopleCnt;

  int reportStatsInterval;

  // per interval stats
  int frames; // for averaging
  int peopleSum;
  int maxPeople;
  int minPeople;
  int moderatorDownWorryTime; 
  int reportStatsTime;

  WorldStats();
  void add(PersonStats *ps) { peopleCnt++;}
  void remove(PersonStats *ps) {peopleCnt--;}
  void update(PersonStats *ps) {}
  void update(int curTime, int deltaTime);
  void reportStats();


  void reset();
}
;
#endif