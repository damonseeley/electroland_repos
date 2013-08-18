#ifndef __GREEDY_TRACKER__
#define __GREEDY_TRACKER__

#include "tracker.h"
#include "TrackMatch.h"

#include <hash_set>
#include <vector>

class GreedyTracker :
	public Tracker
{
public:

	std::vector<Track*> oldTracks;
	std::vector<TrackMatch> matches;

	GreedyTracker(float maxDistSqr);
	~GreedyTracker(void);


	void updateTracks(std::vector<Blob> &blobs, long curtime, long lasttime);
	void GreedyTracker::printTracks();
};


#endif

