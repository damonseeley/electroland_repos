#include "GreedyTracker.h"

GreedyTracker::GreedyTracker(float maxDistSqr) : Tracker( maxDistSqr)
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
void GreedyTracker::updateTracks(std::vector<Blob> &blobs, long curtime, long lasttime) {
	matches.clear();
	oldTracks.clear();


//	std::cout << "----- start ------" << std::endl;
//	printTracks();


	// generate list of all potential matches
	// complexity is blob cnt * track cnt
	// would be better to do some sorting or binning and dis allow matches that are too far away from eachother
	for(std::vector<Blob>::iterator  blobIt = blobs.begin(); blobIt != blobs.end(); ++blobIt) {
		for(std::vector<Track*>::iterator trackIt = tracks.begin(); trackIt != tracks.end(); ++trackIt) {
			Track* t = *trackIt;
			t->isMatched = false;
			if(t->health > 0) { 
				// if the health was 0 it needs to be exited
				matches.push_back(TrackMatch(t,&(*blobIt)));
			}
		}

	}


	oldTracks.swap(tracks);

	// sort so closest possible matches are first
	std::sort(matches.begin(), matches.end());

	// greedy matching, assume the first/closest match found is best
	float scaledMaxDistSqr = maxDistSqr * (curtime-lasttime);

	for(std::vector<TrackMatch>::iterator it = matches.begin(); it != matches.end(); ++it) {
		if(it->matchQaulity > scaledMaxDistSqr) break;// path threshold no more matches
		if(! (it->track->isMatched || it->blob->isMatched)) {  // if both blob and track are free
			it->track->update(it->blob, curtime);
			tracks.push_back(it->track);
		}

	}




	
	// check unmatched tracks
	// cull ones who have been unmached for too long
	// update others and keep around

	for(std::vector<Track *>::iterator it = oldTracks.begin(); it != oldTracks.end(); ++it) {
		Track* t = *it;
		if(! t->isMatched) { // if not already matched (else it was already takencare of above)
			if(t->health <= 0) {
				delete t;
			} else {
				t->updateNoMatch(curtime);
				tracks.push_back(t);
			}
		}
	}


	// create new tracks for unmatched blobs
	for(std::vector<Blob>::const_iterator  it = blobs.begin(); it != blobs.end(); ++it) {
		if(! it->isMatched) { // create a provisional track
			Track *t = new Track(it->x, it->z, curtime);
			tracks.push_back(t);
		}
	}

}