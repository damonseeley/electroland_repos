#include "GreedyTracker.h"

GreedyTracker::GreedyTracker(float maxDistSqr, long provisionalTime, long timeToDeath) : Tracker( maxDistSqr,  provisionalTime,  timeToDeath)
{
}


GreedyTracker::~GreedyTracker(void)
{
} 




//	hash_set<Track> unmatchedTracks;
//	hash_set<Blob> unmatchedKeypoints;

void GreedyTracker::printTracks() {
		std::cout << "tracks: " << std::endl;
	for(std::vector<Track *>::iterator it = tracks.begin(); it != tracks.end(); ++it) {
		std::cout << "     " << (*it)->id << " ismatched: " << (*it)->isMatched  <<std::endl;

	}

	std::cout << "oldTracks: " << std::endl;
	for(std::vector<Track *>::iterator it = oldTracks.begin(); it != oldTracks.end(); ++it) {
		std::cout << "     " << (*it)->id << " ismatched: " << (*it)->isMatched  <<std::endl;
	}

}
void GreedyTracker::updateTracks(std::vector<Blob> &blobs, long curtime) {
	matches.clear();
	oldTracks.clear();
	enters.clear();
	
	for(std::vector<Track *>::iterator it = exits.begin(); it != exits.end(); ++it) {
		Track* t = *it;
		delete t;
	}
	exits.clear();


//	std::cout << "----- start ------" << std::endl;
//	printTracks();


	for(std::vector<Blob>::iterator  blobIt = blobs.begin(); blobIt != blobs.end(); ++blobIt) {
		for(std::vector<Track*>::iterator trackIt = tracks.begin(); trackIt != tracks.end(); ++trackIt) {
			Track* t = *trackIt;
			t->isMatched = false;
			matches.push_back(TrackMatch(t,&(*blobIt)));
		}

	}

	oldTracks.swap(tracks);

//	std::cout << "----- post swap ------" << std::endl;
//	printTracks();

	std::sort(matches.begin(), matches.end());
	for(std::vector<TrackMatch>::iterator it = matches.begin(); it != matches.end(); ++it) {
		if(it->matchQaulity > this->maxDistSqr) break;// no more matches
		if(! (it->track->isMatched || it->blob->isMatched)) {
			it->track->update(it->blob, curtime);
			tracks.push_back(it->track);
		}

		// should I used a set to keep track of unmatched or just iterate over matches (again)
		// get unmatched blobs (make new tracks)
		// get unmatched tracks (cull if needed)
	}


//	std::cout << "----- post match ------" << std::endl;
//	printTracks();


	//alive 
	long deathCutoff = curtime - timeToDeath;
	long provisionalDeathCutoff = curtime - (timeToDeath * .5);




	for(std::vector<Track *>::iterator it = oldTracks.begin(); it != oldTracks.end(); ++it) {
		Track* t = *it;
		if(! t->isMatched) { // if not already matched (else it was already takencare of above)
			if(t->isProvisional) {
				if(t->lastTrack > provisionalDeathCutoff) {
					tracks.push_back(t); 
				} else {
					exits.push_back(t);
				}
			} else {
				if(t->lastTrack > deathCutoff) {
					tracks.push_back(t); 
				} else {
					delete t;
				}
			}
		}
	}

//	std::cout << "----- post exist ------" << std::endl;
//	printTracks();


	for(std::vector<Blob>::const_iterator  it = blobs.begin(); it != blobs.end(); ++it) {
		if(! it->isMatched) { // create a provisional track
			Track *t = new Track(it->x, it->z, curtime);
			enters.push_back(t);
			tracks.push_back(t);
		}
	}

//	std::cout << "----- end  ------" << std::endl;
//	printTracks();

}