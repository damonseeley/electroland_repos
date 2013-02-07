#ifndef __AMBCOMPOSER_H__
#define __AMBCOMPOSER_H__

#include <vector>
#include "Ambient.h"
#include "IGeneric.h"
#include "AmbientCreator.h"




class AmbComposer : public Ambient {
public:
	vector<Ambient *>ambVec;

	virtual void updateFrame(WorldStats *worldStats, int ct, int dt, Interpolators *interps) ;
	int compDelay;
	int timeLeft;
	vector<string> ambNameLs;


	
	AmbComposer(bool ci) ;
	~AmbComposer();

} 
;
class AmbCComposer : public AmbientCreator {
public:
	AmbCComposer(void) {};
	~AmbCComposer(void) {};
	
	virtual Ambient* create(bool ci) {
		return new AmbComposer(ci);
	}
	virtual string getName() { return "Composer"; };

};
#endif