package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.blobtracking.TrackResults;

public class TrackUDPBroadcaster extends UDPBroadcaster implements TrackListener {

	public TrackUDPBroadcaster(int port) throws SocketException, UnknownHostException {
		this("localhost", port);
	}

	public TrackUDPBroadcaster(String address, int port) throws SocketException, UnknownHostException {
		super(address, port);
	}

		
	@Override
	public void updateTracks(TrackResults results) {
		this.send(results);		
	}
	
	

}
