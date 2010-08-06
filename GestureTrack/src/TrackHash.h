#ifndef __TRACK_HASH_H__
#define __TRACK_HASH_H__

#include <map>
#include <vector>
#include <sstream>
#include <iostream>


using namespace std;

class Track {
public:
	unsigned long id;
	float x;
	float y;
	float height;
};

class TrackHash {
public:
	ostringstream msg ;

	vector<unsigned long> enters;
	vector<unsigned long> exits;
	map<unsigned long, Track*> hash;

	TrackHash();
	void clear();
	void addEnter(unsigned long id);
	void addExit(unsigned long id);
	void updateTrack(unsigned long id, float x, float y, float h);

	const char* toString();
};
#endif