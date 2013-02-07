

#ifndef __AMBTRON_H__
#define __AMBTRON_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"
#include "StringSeq.h"

#include <list>



class AmbTron  : public Ambient {
public:
	bool isRunning;

	int stabLength;
	int timeLeftForStab;

	list<BasePixel*> points;
	vector<BasePixel *> stabPoints;
	int curP, curC, curR;
	int gP, gC, gR;
	bool gIsPause;
	int msPerSquare;
	int curIndex;
	int timeLeft;
	bool expanding;
	int tailLength;

	int *path;
	StringSeq *snds;
	int r,g,b;


  AmbTron(bool ci) ;
  ~AmbTron();

  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);
  bool atLoc(int p, int c, int r);
  void moveToLoc(int p, int c, int r);
  void advanceGoalPoint();

} 
;
class AmbCTron : public AmbientCreator {
public:
	AmbCTron(void) {};
	~AmbCTron(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbTron(ci);
	}
	virtual string getName() { return "TargetTron"; };

};
#endif