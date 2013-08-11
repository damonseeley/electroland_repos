#ifndef __TRACK_MATCH__
#define __TRACK_MATCH__
#include "Blob.h"
#include "Track.h"

class TrackMatch
{
public:
	float matchQaulity;
	Track* track;
	Blob* blob;
	TrackMatch(Track* track, Blob* blob);
	bool operator<(const TrackMatch& tm);

	~TrackMatch(void);
};

#endif