#ifndef __AMBSOUND_H__
#define __AMBSOUND_H__

#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"
#include <string>
#include <list>


class SoundCue {
public:
	string snd;
	int delayBeforeSound;
	SoundCue(int t, string s) { snd = s; delayBeforeSound = t; }
}
;
class AmbSound : public Ambient {
public:
	list<SoundCue*> cues;
	int timeLeft;
	string nextSound;
	
  AmbSound(bool ci) ;
  ~AmbSound();
  virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps);
} 
;
class AmbCSound : public AmbientCreator {
public:
	AmbCSound() {};
	~AmbCSound() {};
	
	virtual Ambient* create(bool ci) {
		return new AmbSound(ci);
	}
	virtual string getName() { return "Sound"; };

};
#endif