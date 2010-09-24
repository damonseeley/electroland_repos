#include "ThreadedTrackSender.h"


ThreadedTrackSender::ThreadedTrackSender( UDPSender *udpsender) {
	this->udpsender = udpsender;

	curTrackHash = new TrackHash();
	sendingTrackHash = new TrackHash();


	if(udpsender) {
		sendBarrier = new boost::barrier(2);
		sender = new TrackSender(sendBarrier, this);
	}

}

void ThreadedTrackSender::start() {
	if(udpsender) {
			sender->start();
			sendBarrier->wait();
	}

}

// copy track and return
void ThreadedTrackSender::setCurrentHash(TrackHash* hash) {
	hash->deepCloneHash(curTrackHash);
	sendBarrier->wait();
}


void ThreadedTrackSender::send() {
	sendBarrier->wait();
	swap(curTrackHash, sendingTrackHash);
	char* c = sendingTrackHash->getString();
	udpsender->sendString(c);
}