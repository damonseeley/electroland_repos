#include "Track.h"

long Track::nextId = 0;
float Track::provisionalPenalty = 2;
long Track::provisionalTime = 1000;



//something to make it non provisional here or in tracker
Track::Track(float x, float z, long time) {
	this->x = x;
	this->z = z;
	this->timeToNotProvisional = time + provisionalTime;
	this->lastTrack = time;
	this->id = nextId++;
	this->isMatched = false;
	this->isProvisional = true;
}

float Track::matchQuality(float x, float z) {
	float dx = x-this->x;
	float dz = z-this->z;
	if(isProvisional) {
		return dx*dx + dz * dz + provisionalPenalty;
	}else {
		return dx*dx + dz * dz;
	}
}

void Track::update(Blob *b, long curtime) {
	lastTrack = curtime;
	if(isProvisional) {
		provisionality =  ((float) (timeToNotProvisional-curtime))/ (float) provisionalTime;
		if(provisionality <= 0) {
			provisionality = 0;
			isProvisional= false;
		}
	}
	// do I want smoothign here?
	x = (x + b->x) * .5f;
	z = (z + b->z) * .5f;
	isMatched = true;
	b->isMatched = true;
	health = 1.0;
}
