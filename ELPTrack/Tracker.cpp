#include "Tracker.h"

//	public float maxDistSqr; // max distance between tracks considered a valid move in units/sec
//	public long provisionalTime;
//	public long timeToDeath;

	Tracker::Tracker(float maxDistSqr, long provisionalTime, long timeToDeath) {
		this->maxDistSqr = maxDistSqr;
		this->provisionalTime = provisionalTime;
		this->timeToDeath = timeToDeath;
		Track::provisionalTime = provisionalTime;
	}
//void Tracker::updateTracks(std::vector<cv::KeyPoint> *keypoints) {
//}