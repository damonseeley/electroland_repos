#ifndef __OSC_TRACK_SENDER__
#define __OSC_TRACK_SENDER__

#include <string>
#include <vector>
#include "Tracker.h"
#include "ip/UdpSocket.h"


#define OUTPUT_BUFFER_SIZE 4096

class OSCTrackSender {

public:

	float oscMinX;
	float oscMaxX;
	float oscMinZ;
	float oscMaxZ;
	
	float worldMinX;
	float worldMaxX;
	float worldMinZ;
	float worldMaxZ;

	float scaleX;
	float scaleZ;

	UdpTransmitSocket *transmitSocket;
	char buffer[OUTPUT_BUFFER_SIZE];

	OSCTrackSender(std::string ip, int port);	
	void sendTracks(Tracker *tracker);

	void setTransform(
		float oscMinX, float oscMaxX,
		float oscMinZ, float oscMaxZ,	
		float worldMinX, float worldMaxX,
		float worldMinZ, float worldMaxZ);	

};

#endif
