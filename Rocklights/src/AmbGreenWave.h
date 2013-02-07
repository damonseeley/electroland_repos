#ifndef __AMBGREENWAVE_H__
#define __AMBGREENWAVE_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"


class AmbGreenWave : public Ambient {
	int msPerStick;
	int curStick;
	int timeLeftForStick;
	int stepSize;
	int width;
	int maxSize;
	int curWave;
public:
	string colString;
	int r, g, b;
	static int curCols[MAXWAVES];
	static int curWaveCnt;

public:
	AmbGreenWave(bool ci);
	~AmbGreenWave(void);
	void resetParams();
	 virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) ;

};



class AmbCGreenWave : public AmbientCreator {
public:
	AmbCGreenWave(void) {};
	~AmbCGreenWave(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbGreenWave *a =  new AmbGreenWave(ci);
		a->r = 0.0f;
		a->g = 1.0f;
		a->b = 0.0f;
		a->colString = "Green";
		a->resetParams();
		return a;
	}
	virtual string getName() { return "GreenWave"; };

};

class AmbCRedWave : public AmbientCreator {
public:
	AmbCRedWave(void) {};
	~AmbCRedWave(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbGreenWave *a =  new AmbGreenWave(ci);
		a->r = 1.0f;
		a->g = 0.0f;
		a->b = 0.0f;
		a->colString = "Red";
		a->resetParams();
		return a;
	}
	virtual string getName() { return "RedWave"; };

};

class AmbCBlueWave : public AmbientCreator {
public:
	AmbCBlueWave(void) {};
	~AmbCBlueWave(void) {};
	
	virtual Ambient* create(bool ci) {
		AmbGreenWave *a =  new AmbGreenWave(ci);
		a->r = 0.0f;
		a->g = 0.0f;
		a->b = 1.0f;
		a->colString = "Blue";
		a->resetParams();
		return a;
	}
	virtual string getName() { return "BlueWave"; };

};

#endif;