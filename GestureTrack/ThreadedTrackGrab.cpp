#include "ThreadedTrackGrab.h"


ThreadedTrackGrab::ThreadedTrackGrab(PersonTrackReceiver *receiver, UDPSender *udpsender) {
	this->receiver = receiver;
	this->udpsender = udpsender;

	grabbedTrackHash = new TrackHash();
	curTrackHash = new TrackHash();
	sendingTrackHash = new TrackHash();


	if(receiver) {
	grabBarrier = new boost::barrier(2);
	grabber = new TrackGrabber(grabBarrier, this);
	}
	if(udpsender) {
	sendBarrier = new boost::barrier(2);
	sender = new TrackSender(sendBarrier, this);
}



}

void ThreadedTrackGrab::start() {
	if(receiver) {
			receiver->start();
			grabber->start();
			grabBarrier->wait(); // wait for inital grab
	}
	if(udpsender) {
			sender->start();
			sendBarrier->wait();
	}

}

// swap grab with cur return cur
TrackHash* ThreadedTrackGrab::getCurrentHash() {
	if(receiver) {
	TrackHash *tmp = grabbedTrackHash;
	grabbedTrackHash = curTrackHash;
	curTrackHash = tmp;
	grabBarrier->wait();
	} else {
		curTrackHash->clear();
	}
	return curTrackHash;

}

// swap cur with send send cur
void ThreadedTrackGrab::trackHashUpdated(){
	if(udpsender) {
	TrackHash *tmp = sendingTrackHash;
	sendingTrackHash = curTrackHash;
	curTrackHash = tmp;

	sendBarrier->wait();
	}
}
void ThreadedTrackGrab::grab() {
	
//	std::cout << "before grab" <<std::endl;
	DWORD before =  timeGetTime();
	receiver->grab(grabbedTrackHash);
	DWORD after =  timeGetTime();
//	std::cout << "after grab time = " << (after-before) << std::endl;
}

void ThreadedTrackGrab::send() {
	char* c = sendingTrackHash->getString();
	udpsender->sendString(c);
}