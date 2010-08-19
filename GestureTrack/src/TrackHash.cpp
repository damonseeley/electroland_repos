#include "TrackHash.h"
#include <GL/glut.h>


TrackHash::TrackHash() {
	msgCharSize = 0;
 msgChar = new char[0];
}


void TrackHash::clearEnterAndExits() {
	enters.clear();
	exits.clear();
}
void TrackHash::clear() {
	enters.clear();
	exits.clear();

	for(map<unsigned long, Track*>::iterator i = hash.begin(); i != hash.end(); i++)
	{
		Track *t = i->second;
		delete t;
	}
	hash.clear();
}

void TrackHash::addTrack(Track* t) {
	hash[t->id] = t;
}

void TrackHash::addEnter(unsigned long id) {
//	std::cout <<"enter " << id << std::endl;
//	if(! id) return;
//	enters.push_back(id);
//	Track *track = new Track();
//	track->id = id;
//	hash[id]= track;
}
void TrackHash::addExit(unsigned long id) {
//	std::cout <<"exit " << id << std::endl;
//	if(! id) return;
//	exits.push_back(id);
//	Track *t = hash[id];
//	if(t) {
//		hash.erase(id);
//		delete t;
//	}
}


void TrackHash::updateTrack(unsigned long id, float x, float y, float h, long validUntillFrame) {
	if(! id) return;
	Track *t = hash[id];
	if(! t) {
		t = new Track();
		t->id = id;
		hash[id] = t;
	}
	t->updateTyzxTrack(x,y,h, validUntillFrame);
//	t->x = x;
//	t->z = y; // what the tracking system calls y we call z
//	t->height = h;
	

//	} else {
//		std::cout <<"track w/o enter " << id << std::endl;
//		addEnter(id);
//		updateTrack(id,x,y,h);
//	}
}


char* TrackHash::getString() {
	ostringstream msg ;
	msg << fixed;
	msg.precision(3);
	
	
	for(map<unsigned long, Track*>::iterator i = hash.begin();
		i != hash.end();
		i++)
	{
		Track *t = i->second;
		if(t) {
			t->buildString(msg);
			msg << ", " << endl;
		}

	}

//	msg << "end of msg" << endl;

	string msgStr = msg.rdbuf()->str();


	if(msgStr.size() >= msgCharSize) {
		delete msgChar;
		msgCharSize = msgStr.size() + 1;
		msgChar = new char[msgCharSize];
	}
	strcpy(msgChar, msgStr.c_str());

	return msgChar;
}

void TrackHash::render() {
	glColor4f(0.0f, 1.0f,1.0f, .90f);
	for(map<unsigned long, Track*>::iterator i = hash.begin();
		i != hash.end();
		i++)
	{

		Track *t = i->second;
		if(t) {
			glPushMatrix();
//			glRotatef(90.0f,1.0f,0.0f,0.0f);
			
			glTranslatef(t->x->value, t->height->value, t->z->value);
//			std::cout << " rendering at " << t->x->value << " , " <<  t->center->value << " , " << t->z->value << std::endl;
			glutWireSphere(.1, 5,5);
//			glutSolidCone(.25f, t->height->value, 6, 2);

			glPopMatrix();
		}
	}


}
void TrackHash::updateValidity(long curFrame) {
	for(map<unsigned long, Track*>::iterator i = hash.begin();
		i != hash.end();
		i++)
	{

		Track *t = i->second;
		if(t) {
			t->updatePropValidity(curFrame);
		}
	}
}


// gonna do a stupid simple find the closes free match
void TrackHash::merge(TrackHash *otherHash, float maxDistSqr, long curFrame) {
	matchedTracks.clear();

	for(map<unsigned long, Track*>::iterator i = otherHash->hash.begin(); i != otherHash->hash.end(); i++)
	{
		Track *otherTrack = i->second;
		if(otherTrack) {
			Track* closest = NULL;
			float closestDistSqr = maxDistSqr;
			for(map<unsigned long, Track*>::iterator j = hash.begin(); j != hash.end(); j++)
			{
				Track *localTrack = j->second;
				if(localTrack) {
					float d = localTrack->distSqr(otherTrack);
					if (d < closestDistSqr) {
						closestDistSqr = d;
						closest = localTrack;
					}
				}
			}
			if(closest != NULL) {
				hash.erase(closest->id);
				closest->merge(otherTrack, curFrame);
				matchedTracks.push_back(closest);

			} else {
				matchedTracks.push_back(otherTrack);
			}
		}
		
	}
		for(int i = 0; i < matchedTracks.size(); i++) {
			Track *t = matchedTracks[i];
			if(t) {
			otherHash->hash.erase(t->id); // remove from other hash so not deleted on clear
			hash[t->id] = t;
			}
		}
	
}
