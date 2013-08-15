#include "OSCTrackSender.h"
#include "osc/OscOutboundPacketStream.h"
#include "boost\lexical_cast.hpp"


void OSCTrackSender::setTransform(
		float oscMinX, float oscMaxX,
		float oscMinZ, float oscMaxZ,	
		float worldMinX, float worldMaxX,
		float worldMinZ, float worldMaxZ) {
			this->oscMinX = oscMinX;
			this->oscMaxX = oscMaxX;
			this->oscMinZ = oscMinX;
			this->oscMaxZ = oscMaxZ;
			this->worldMinX = worldMinX;
			this->worldMaxX = worldMaxX;
			this->worldMinZ = worldMinZ;
			this->worldMaxZ = worldMaxZ;

			scaleX = (oscMaxX - oscMinX) / (worldMaxX - worldMinX);
			scaleZ = (oscMaxZ - oscMinZ) / (worldMaxZ - worldMinZ);
	}

OSCTrackSender::OSCTrackSender(std::string ip, int port) {
	if(ip.empty()) { // dont' create socet if empty
		transmitSocket = NULL;
	} else {
		transmitSocket = new UdpTransmitSocket( IpEndpointName( ip.c_str(), port ) );
	}
	// no scaling by default
	scaleX = 1.0f;
	scaleZ = 1.0f;
	worldMinX = 0;
	worldMinZ = 0;
	oscMinX = 0;
	oscMinZ = 0;

	


}

void OSCTrackSender::sendTracks(Tracker *tracker) {
	if(! transmitSocket) return; // if no socket skip 
	osc::OutboundPacketStream oscStream(buffer, OUTPUT_BUFFER_SIZE);
	oscStream << osc::BeginBundleImmediate;

	oscStream << osc::BeginMessage("/metaInfo");
	oscStream << oscMinX << oscMaxX << oscMinZ << oscMaxZ;
	oscStream << osc::EndMessage;

	oscStream << osc::BeginMessage("/tracks");
	for(std::vector<Track*>::iterator it = tracker->tracks.begin(); it != tracker->tracks.end(); it++) {
		Track *t = *it;
		float x = (t->x - worldMinX) * scaleX;
		float z = (t->z - worldMinZ) * scaleZ;
		x = (x < oscMinX) ? oscMinX : x;
		z = (z < oscMinZ) ? oscMinZ : z;
		x = (x > oscMaxX) ? oscMaxX : x;
		z = (z > oscMaxZ) ? oscMaxZ : z;
		oscStream << t->id << x << z  << t->provisionality << t->health;
	}
	oscStream << osc::EndMessage;

	oscStream << osc::BeginMessage("/enters");
	for(std::vector<Track*>::iterator it = tracker->enters.begin(); it != tracker->enters.end(); it++) {
		Track *t = *it;
		float x = (t->x - worldMinX) * scaleX;
		float z = (t->z - worldMinZ) * scaleZ;
		x = (x < oscMinX) ? oscMinX : x;
		z = (z < oscMinZ) ? oscMinZ : z;
		x = (x > oscMaxX) ? oscMaxX : x;
		z = (z > oscMaxZ) ? oscMaxZ : z;
		oscStream << t->id << x << z  << t->provisionality << t->health;
	}
	oscStream << osc::EndMessage;

	oscStream << osc::BeginMessage("/exits");
	for(std::vector<Track*>::iterator it = tracker->exits.begin(); it != tracker->exits.end(); it++) {
		Track *t = *it;
		float x = (t->x - worldMinX) * scaleX;
		float z = (t->z - worldMinZ) * scaleZ;
		x = (x < oscMinX) ? oscMinX : x;
		z = (z < oscMinZ) ? oscMinZ : z;
		x = (x > oscMaxX) ? oscMaxX : x;
		z = (z > oscMaxZ) ? oscMaxZ : z;
		oscStream << t->id << x << z  << t->provisionality << t->health;
	}
	oscStream << osc::EndMessage;

	oscStream << osc::EndBundle;
	transmitSocket->Send(oscStream.Data(), oscStream.Size());

}