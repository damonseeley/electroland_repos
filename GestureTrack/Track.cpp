#include "Track.h"
#include <sstream>
//	void merge(TrackProp *updateProp, float weight); 
	// if both are valid weighted average and validUntilFrame is updateProps validUntilFrame
	// if this is not valid full value of updateProp (if it is valid)
	// if updateProp is not valid and this is no change


void TrackProp::merge(TrackProp *updateProp, float weight, long curFrame) {
	if(updateProp->isValid(curFrame)) {
		if(this->isValid(curFrame)) {
			// both are valid take weighted average
			this->value *= 1.0f - weight;
			this->value += updateProp->value * weight;
			this->validUntillFrame = (this->validUntillFrame  > updateProp->validUntillFrame) ? this->validUntillFrame  : updateProp->validUntillFrame ;
		} else { // this is not valid so use new value
			this->value = updateProp->value;
			this->validUntillFrame = updateProp->validUntillFrame;
		}
	} 
}
void TrackProp::buildString(ostringstream &msg){
	if(this->isValidBool) {
		msg << name << " : " <<value << ";" << std::endl;
	}
}


Track::Track() {
	x = new TrackProp("x");
	z = new TrackProp("z");
	height = new TrackProp("h");
	center = new TrackProp("center");


	lhX = new TrackProp("lhX");
	lhY = new TrackProp("lhY");
	lhZ = new TrackProp("lhZ");

	rhX = new TrackProp("rhX");
	rhY = new TrackProp("rhY");
	rhZ = new TrackProp("rhZ");

	jump = new TrackProp("jump");
	activity = new TrackProp("activity");
}
float Track::distSqr(Track *t) {
	float dx = t->x->value - x->value;
	float dz = t->z->value - z->value;
	return dx*dx + dz*dz;

}

void Track::updatePropValidity(long curFrame) {
	x->setIsValid(curFrame);
	z->setIsValid(curFrame);
	height->setIsValid(curFrame);
	center->setIsValid(curFrame);
	lhX->setIsValid(curFrame);
	lhY->setIsValid(curFrame);
	lhZ->setIsValid(curFrame);
	rhX->setIsValid(curFrame);
	rhY->setIsValid(curFrame);
	rhZ->setIsValid(curFrame);
	jump->setIsValid(curFrame);
	activity->setIsValid(curFrame);

}

void Track::updateTyzxTrack(float x, float z, float h, long validUntillFrame) {
	this->x->updateValue(x, validUntillFrame);
	this->x->isValidBool = true;
	this->z->updateValue(z, validUntillFrame);
	this->z->isValidBool = true;
	this->height->updateValue(h, validUntillFrame);
	this->height->isValidBool = true;
}

void Track::merge(Track *track, long curFrame) {
	x->merge(track->x, .5f, curFrame);
	z->merge(track->z, .5f, curFrame);
	height->merge(track->height, .5f, curFrame);

	// jump is the diff between the current center and the slow moving average center
	// so calculate it during merge before updating center
	if(center->isValid(curFrame) && track->center->isValid(curFrame)) {
		track->jump->value = track->center->value - center->value;
		track->jump->validUntillFrame = curFrame + 2;
		track->jump->isValidBool = true;
	} 
	

	center->merge(track->center, .05f, curFrame); // update smoother than most qualities
	jump->merge(track->jump, .75, curFrame); // instant is more important
	lhX->merge(track->lhX, .5f, curFrame);
	lhY->merge(track->lhY, .5f, curFrame);
	lhZ->merge(track->lhZ, .5f, curFrame);
	rhX->merge(track->rhX, .5f, curFrame);
	rhY->merge(track->rhY, .5f, curFrame);
	rhZ->merge(track->rhZ, .5f, curFrame);
	activity->merge(track->activity, .5f, curFrame);

	lastUpdated = curFrame;

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


