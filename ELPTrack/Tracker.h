#ifndef __TRACKER__
#define __TRACKER__
#include <opencv2/features2d/features2d.hpp>
#include <vector>
#include "Track.h"
#include "Blob.h"

class Tracker {
public:
	 float maxDistSqr; // max distance between tracks considered a valid move in units/sec
	 long provisionalTime;
	 long timeToDeath;

	 std::vector<Track*> tracks;
	 	std::vector<Track*> enters;
	std::vector<Track*> exits;

	Tracker(float maxDistSqr, long provisionalTime, long timeToDeath);
 	virtual void updateTracks(std::vector<Blob> &blobs, long curtime) = 0;
};

#endif