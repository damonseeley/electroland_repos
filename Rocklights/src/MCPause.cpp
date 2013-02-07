#include "mcpause.h"
#include "AmbientCreatorHash.h"
#include "MCGeneric.h"

MCPause::MCPause(PeopleStats *peopleStats, string to, string from, int time):MasterController(peopleStats) {
	pauseTime = time;
	toName = to;
	fromName = from;
	curAmbient = AmbientCreatorHash::theACHash->create("AmbEmpty");
	peopleStats->transitionTo(new Pattern(), (int) (time * .25));
	name = "";
	timeToIntermission = -1;
}

void MCPause::updateFrame(WorldStats *worldStats, PeopleStats *ps, int ct, int dt) {
	if(pauseTime > 0) {
		pauseTime-=dt;
	} else {
		new MCGeneric(ps, toName, fromName);
	}
}

