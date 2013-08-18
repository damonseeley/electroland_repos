#ifndef __TRACK__
#define __TRACK__

#include "Blob.h"
#include <iostream>
#include "osc/OscOutboundPacketStream.h"

class Track {

private:
	static float smoothing;
	static float smoothingInv;

	void update(long curtime); // update for blob and no blob




public:
	static long nextId;
	static float provisionalPenalty;
	static long provisionalTime;
	static long provisionalTimeToDeath;
	static long timeToDeath;

	

	long id;

	float x;
	float z;

	long timeToNotProvisional;
	long lastTrack;
	long establishmentTime;

	bool isMatched;
	bool isProvisional;

	float provisionality;
	float health;
	long age;

	Track(float x, float z, long time);
	
	float matchQuality(float x, float z); // basically distance (lower is better)

	void update(Blob *b, long curtime);
	void updateNoMatch(long curtime);
	friend std::ostream& operator<<(std::ostream& os, const Track& t);
	static void setSmoothing(float f);

};



#endif
