#ifndef __TRACK_H__
#define __TRACK_H__

#include <string>
#include <sstream>
#include <iostream>

using namespace std;
class TrackProp {
public:
	bool isValidBool;
	string name;
	long validUntillFrame;
	float value;

	TrackProp(string name) { this->name = name; validUntillFrame = -1; }
	void updateValue(float v, int validUntillFrame) { value = v; this->validUntillFrame = validUntillFrame; }
	bool isValid(long curFrame) { return curFrame <= validUntillFrame; }
	void setIsValid(long curFrame) { isValidBool = isValid(curFrame); }

	void merge(TrackProp *updateProp, float weight, long curFrame); 

	void buildString(ostringstream &msg);
};


class Track {
public:


	long lastUpdated; 

	unsigned long id;

	// from tracking system
	TrackProp *x;
	TrackProp *z;
	TrackProp *height;
	TrackProp *center;

	//left hand
	TrackProp *lhX;
	TrackProp *lhY;
	TrackProp *lhZ;

	// right hand
	TrackProp *rhX;
	TrackProp *rhY;
	TrackProp *rhZ;

	TrackProp *jump;
	TrackProp *activity;

	Track();


	void updateTyzxTrack(float x, float z, float h, long validUntillFrame);

	void updatePropValidity(long curFrame);

	void merge(Track *track, long curFrame);

	void buildString(ostringstream &msg);

	float distSqr(Track *t);

};

#endif