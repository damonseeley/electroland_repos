#include "Tracker.h"

//	public float maxDistSqr; // max distance between tracks considered a valid move in units/sec
//	public long provisionalTime;
//	public long timeToDeath;

	Tracker::Tracker(float maxDistSqr) {
		this->maxDistSqr = maxDistSqr;
	}
//void Tracker::updateTracks(std::vector<cv::KeyPoint> *keypoints) {
//}