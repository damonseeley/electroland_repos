#include "TrackHash.h"
#include <GL/glut.h>
#include "Tyzx3x3Matrix.h"


TrackHash::TrackHash() {
		rotationX =0;
	rotationZ = 0;
	rotationAngle=0;
	sinAngle=0;
	cosAngle=0;

	msgCharSize = 0;
 msgChar = new char[0];
}


void TrackHash::clearEnterAndExits() {
	enters.clear();
	exits.clear();
}

void TrackHash::deepCloneHash(TrackHash *other) {
	other->clear();
	for(map<unsigned long, Track*>::iterator i = hash.begin(); i != hash.end(); i++) {
		Track *t = i->second;
		other->hash[t->id] = new Track(t);
	}

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



// gonna do a stupid simple find the closes free match
void TrackHash::merge(TrackHash *otherHash, float maxDistSqr, long curFrame) {
	matchedTracks.clear();

	// cull old tracks first
	for(map<unsigned long, Track*>::iterator i = hash.begin(); i != hash.end(); ) {
		Track *t = i->second;
		if(t->culltime < curFrame) {
			hash.erase(i++);
		} else {
			i++;

		}
	}



	for(map<unsigned long, Track*>::iterator i = otherHash->hash.begin(); i != otherHash->hash.end();)
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
				hash.erase(closest->id);  // no longer in contention
				closest->merge(otherTrack, curFrame);
				matchedTracks.push_back(closest);
				delete otherTrack;
			} else {
				matchedTracks.push_back(otherTrack);
			}

			otherHash->hash.erase(i++); 
			// either the track was merged and put in matches or it was and put in matchs
			// eitherway we are done with it

			// copy the matchs into existing
			for(int i = 0; i < matchedTracks.size(); i++) {
				Track *t = matchedTracks[i];
				if(t) { // why do I need this
				hash[t->id] = t;
			}
		}


		}
		
	}
	//other tracks should be empty now
	
}

void TrackHash::setRotation(float angle) {
	rotationAngle =angle;
	sinAngle = sinf(rotationAngle / 180.0 * PI);
	cosAngle = cosf(rotationAngle / 180.0 * PI);

}

void TrackHash::applyRotation() {
	for(map<unsigned long, Track*>::iterator i = hash.begin();
		i != hash.end();
		i++)
	{
		Track *t = i->second;
		float x = t->x->value;
		float z = t->z->value;

		if(t) {
			t->x->value = rotationX + (cosAngle * (x - rotationX)) - (sinAngle * (z - rotationZ));
			t->z->value = rotationZ + (sinAngle * (x - rotationX)) + (cosAngle * (z - rotationZ));
		}

	}

}