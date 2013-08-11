#include "TrackMatch.h"


TrackMatch::TrackMatch(Track* track, Blob* blob)
{
	this->track = track;
	this->blob = blob;
	matchQaulity = track->matchQuality(blob->x, blob->z);
}


TrackMatch::~TrackMatch(void)
{

}

	bool TrackMatch::operator<(const TrackMatch& tm) {
		return this->matchQaulity < tm.matchQaulity;
	}