#ifndef __INTERPGEN_H__
#define __INTERPGEN_H__



class InterpGen {
  float deltaPerUSec;
  float perc;
public:
  bool isRunning;
  bool wasStarted;

  InterpGen();
  void start(int timeUSecs);
  void reset(); // sets was started to false
  float update(int deltaT);


}
;
#endif
