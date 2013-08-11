#ifndef __TRACK__
#define __TRACK__

#include "Blob.h"
#include <iostream>

class Track {
public:
	static long nextId;
	static float provisionalPenalty;
	static long provisionalTime;

	long id;

	float x;
	float z;

	long timeToNotProvisional;
	long lastTrack;

	bool isMatched;
	bool isProvisional;

	Track(float x, float z, long time);
	
	float matchQuality(float x, float z); // basically distance (lower is better)

	void update(Blob *b, long curtime);


};

#endif
