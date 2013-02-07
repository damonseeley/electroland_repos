#ifndef __MCPAUSE_H__
#define __MCPAUSE_H__

#include <string>
#include "MasterController.h"

class MCPause : public MasterController {
	int pauseTime;
	string fromName;
	string toName;
public:
	MCPause(PeopleStats *peopleStats, string nextName, string fromName, int time);
	virtual void updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt);
};
#endif