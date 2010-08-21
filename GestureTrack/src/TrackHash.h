#ifndef __TRACK_HASH_H__
#define __TRACK_HASH_H__

#include <map>
#include <vector>
#include <sstream>
#include <iostream>

#include "Track.h"



using namespace std;

class TrackHash {
public:

	int  msgCharSize;
	char* msgChar;
	vector<Track*> matchedTracks;
	vector<unsigned long> enters;
	vector<unsigned long> exits;
	map<unsigned long, Track*> hash;

	float rotationX;
	float rotationZ;
	float rotationAngle;

	float sinAngle;
	float cosAngle;

	TrackHash();
	void clear();
	void clearEnterAndExits();
	void addEnter(unsigned long id);
	void addExit(unsigned long id);
//	void updateTrack(unsigned long id, float x, float y, float h);
	void TrackHash::updateTrack(unsigned long id, float x, float y, float h, long validUntillFrame);
	void addTrack(Track* t);
	
	char* getString();

	//const char* toString();
	void render();

	// merges tra
	void merge(TrackHash *otherHash, float maxDistSqr, long curFrame); 

	void updateValidity(long curFrame);

	void setRotationPoint(float x, float z) { rotationX = x; rotationZ = z; };
	void setRotation(float angle);
	void applyRotation();
};
#endif