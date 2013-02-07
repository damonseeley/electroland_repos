#ifndef __MCGENERIC_H__
#define __MCGENERIC_H__

#include <string>
#include <vector>
#include "AmbientCreatorHash.h"
#include "MasterController.h"
#include "PatGeneric.h"
#include "StringSeq.h"

class MCGeneric : public MasterController {
	string transFromName;
	
	int lowTresh;
	StringSeq *lowTransTols;
	
	int highThresh;
	StringSeq *highTransTols;

	int lifeTime;
	StringSeq *timeTransTols;  // BACK means go back to last state

	string ambientName;

	int afterPauseTime;
	

public:
	MCGeneric(PeopleStats *peopleStats, string name, string transFromName);
	~MCGeneric();
	virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);

};

#endif
