#include "OSCTrackSender.h"
#include "osc/OscOutboundPacketStream.h"
#include <boost\lexical_cast.hpp>


void OSCTrackSender::setTransform(
		float oscMinX, float oscMaxX,
		float oscMinZ, float oscMaxZ,	
		float worldMinX, float worldMaxX,
		float worldMinZ, float worldMaxZ) {

			if((oscMinX == 0) && (oscMaxX ==0)) {
			this->oscMinX = worldMinX;
			this->oscMaxX = worldMaxX;
			this->worldMinX = worldMinX;
			this->worldMaxX = worldMaxX;
			scaleX = 1;
			} else {
			this->oscMinX = oscMinX;
			this->oscMaxX = oscMaxX;
			this->worldMinX = worldMinX;
			this->worldMaxX = worldMaxX;
			scaleX = (oscMaxX - oscMinX) / (worldMaxX - worldMinX);
			}

			if((oscMinZ == 0) && (oscMaxZ ==0)) {
			this->oscMinZ = worldMinZ;
			this->oscMaxZ = worldMaxZ;
			this->worldMinZ = worldMinZ;
			this->worldMaxZ = worldMaxZ;
			scaleZ = 1;
			} else {
			this->oscMinZ = oscMinZ;
			this->oscMaxZ = oscMaxZ;
			this->worldMinZ = worldMinZ;
			this->worldMaxZ = worldMaxZ;
			scaleZ = (oscMaxZ - oscMinZ) / (worldMaxZ - worldMinZ);
			}

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
	oscStream << oscMinX << oscMaxX << oscMinZ << oscMaxZ << worldMinX << worldMaxX << worldMinZ << worldMaxZ << 6; // 6 fields in track data
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
		oscStream << t->id << x << z << (t->age * .001f) << t->provisionality << t->health;
		//std::cout << (*t) << std::endl;

		
	}
	oscStream << osc::EndMessage;

	oscStream << osc::EndBundle;
	transmitSocket->Send(oscStream.Data(), oscStream.Size());

}