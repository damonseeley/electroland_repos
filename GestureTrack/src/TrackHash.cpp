#include "TrackHash.h"

TrackHash::TrackHash() {
		msg << fixed;
	msg.precision(0);
}
void TrackHash::clear() {
	enters.clear();
	exits.clear();
	hash.clear();
}

void TrackHash::addEnter(unsigned long id) {
	enters.push_back(id);
	Track *track = new Track();
	track->id = id;
	hash[id]= track;
}
void TrackHash::addExit(unsigned long id) {
	exits.push_back(id);
	Track *t = hash[id];
	hash.erase(id);
	delete t;
}

void TrackHash::updateTrack(unsigned long id, float x, float y, float h) {
	Track *t = hash[id];
	t->x = x;
	t->y = y;
	t->height = h;
}

const char* TrackHash::toString() {
	msg.clear();
	
	for(map<unsigned long, Track*>::iterator i = hash.begin();
		i != hash.end();
		i++)
	{
		Track *t = i->second;
		msg << t->id << " { " << endl;
		msg << "  " << "x : " << t->x << ";" <<endl;
		msg << "  " << "y : " << t->y << ";" << endl;
		msg << "  " << "h : " << t->height << ";" << endl;
		msg << "}" << endl;

	}
	return msg.rdbuf()->str().c_str();
	msg.seekp(ios_base::beg); // not sure if we can reuse it

}