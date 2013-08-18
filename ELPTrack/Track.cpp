#include "Track.h"

long Track::nextId = 0;
float Track::provisionalPenalty = 2;
long Track::provisionalTime = 1000;
long Track::timeToDeath = 1000;
long Track::provisionalTimeToDeath = (long) (timeToDeath * .5);
float Track::smoothing = .5f;
float Track::smoothingInv = .5f;

//something to make it non provisional here or in tracker
Track::Track(float x, float z, long time) {
	this->x = x;
	this->z = z;
	this-> establishmentTime = -1;
	this->timeToNotProvisional = time + provisionalTime;
	this->lastTrack = time;
	this->id = nextId++;
	this->isMatched = false;
	this->isProvisional = true;
	this->health = 1;
	this->age = 0;
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

void Track::update(long curtime) {
	if(isProvisional) {
		provisionality =  ((float) (timeToNotProvisional-curtime))/ (float) provisionalTime;
		if(provisionality <= 0) {
			provisionality = 0;
			isProvisional= false;
			age = 0;
			establishmentTime = curtime;
		}
		health = 1.0f - ( (curtime-lastTrack) / provisionalTimeToDeath);
	} else {
		age = curtime-establishmentTime;
		health = 1.0f - ( (curtime-lastTrack) / timeToDeath);
	}
	health = (health < 0) ? 0 : health;

}

void Track::updateNoMatch(long curtime) {
	isMatched = false;
	update(curtime);
}

void Track::update(Blob *b, long curtime) {
	lastTrack = curtime;
	isMatched = true;
	b->isMatched = true;
	// do I want smoothing here?
	x = x * Track::smoothing + b->x * Track::smoothingInv;
	z = z * Track::smoothing + b->z * Track::smoothingInv;
	update(curtime);
}
std::ostream& operator<<(std::ostream& os, const Track& t)
{
	os << t.id << ":(" << t.x << ',' << t.z << ") age=" << t.age << " prov=" <<t.provisionality << "health= " << t.health;
	return os;
}

void Track::setSmoothing(float f) {
	f = (f < 0) ? 0 : f;
	f = (f >= 1) ? .99999999999999999f : f;
	Track::smoothing = f;
	Track::smoothingInv = 1.0f-f;
}