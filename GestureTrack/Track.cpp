#include "Track.h"
#include <sstream>

int Track::LIFESPAN = 10;

//	void merge(TrackProp *updateProp, float weight); 
	// if both are valid weighted average and validUntilFrame is updateProps validUntilFrame
	// if this is not valid full value of updateProp (if it is valid)
	// if updateProp is not valid and this is no change


void TrackProp::merge(TrackProp *updateProp, float weight) {
			this->value *= 1.0f - weight;
			this->value += updateProp->value * weight;
}
void TrackProp::buildString(ostringstream &msg){
//	if(this->isValidBool) {
		msg << name << " : " <<value << ";" << std::endl;
//	}
}


Track::Track(Track * t) {
	id = t->id;
	x = new TrackProp("x", t->x->value);
	z = new TrackProp("z", t->z->value);
	height = new TrackProp("h", t->height->value);
}
Track::Track() {
	x = new TrackProp("x");
	z = new TrackProp("z");
	height = new TrackProp("h");
}

float Track::distSqr(Track *t) {
	float dx = t->x->value - x->value;
	float dz = t->z->value - z->value;
	return dx*dx + dz*dz;

}


void Track::updateTyzxTrack(float x, float z, float h, long validUntillFrame) {
	this->x->updateValue(x);
	this->z->updateValue(z);
	this->height->updateValue(h);
}

void Track::merge(Track *track, long curFrame) {
	x->merge(track->x, .5f);
	z->merge(track->z, .5f);
	height->merge(track->height, .5f);
	culltime = curFrame + LIFESPAN;
	// jump is the diff between the current center and the slow moving average center
	// so calculate it during merge before updating center
//	if(center->isValid(curFrame) && track->center->isValid(curFrame)) {
//		track->jump->value = track->center->value - center->value;
//		track->jump->validUntillFrame = curFrame + 2;
//		track->jump->isValidBool = true;
//	} 
	

//	center->merge(track->center, .05f, curFrame); // update smoother than most qualities
//	jump->merge(track->jump, .75, curFrame); // instant is more important
//	lhX->merge(track->lhX, .5f, curFrame);
//	lhY->merge(track->lhY, .5f, curFrame);
//	lhZ->merge(track->lhZ, .5f, curFrame);
//	rhX->merge(track->rhX, .5f, curFrame);
//	rhY->merge(track->rhY, .5f, curFrame);
//	rhZ->merge(track->rhZ, .5f, curFrame);
//	activity->merge(track->activity, .5f, curFrame);

//	lastUpdated = curFrame;

}

void Track::buildString(ostringstream &msg) {
	msg << id << " { " ;
	x->buildString(msg);
	z->buildString(msg);
	height->buildString(msg);
//	center->buildString(msg);
//	lhX->buildString(msg);
//	lhY->buildString(msg);
//	lhZ->buildString(msg);
//	rhX->buildString(msg);
//	rhY->buildString(msg);
//	rhZ->buildString(msg);
//	jump->buildString(msg);
//	activity->buildString(msg);
	msg << "}";
}


