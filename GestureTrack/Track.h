#ifndef __TRACK_H__
#define __TRACK_H__

#include <string>
#include <sstream>
#include <iostream>

using namespace std;
class TrackProp {
public:
	string name;
	float value;
	TrackProp(string name) { this->name = name; }
	TrackProp(string name, float v) { this->name = name; value = v; }
	void updateValue(float v) { value = v;}

	void merge(TrackProp *updateProp, float weight); 

	void buildString(ostringstream &msg);
};


class Track {
public:
	static int LIFESPAN;


	int culltime; 

	unsigned long id;

	// from tracking system
	TrackProp *x;
	TrackProp *z;
	TrackProp *height;
//	TrackProp *center;

	//left hand
//	TrackProp *lhX;
//	TrackProp *lhY;
//	TrackProp *lhZ;

	// right hand
//	TrackProp *rhX;
//	TrackProp *rhY;
//	TrackProp *rhZ;

//	TrackProp *jump;
//	TrackProp *activity;

	Track();
	Track(Track *t);


	void updateTyzxTrack(float x, float z, float h, long validUntillFrame);


	void merge(Track *track, long curFrame);

	void buildString(ostringstream &msg);

	float distSqr(Track *t);

};

#endif