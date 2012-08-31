package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.blobtracking.TrackListener;
import net.electroland.elvis.blobtracking.TrackResults;

public class TrackUPDBroadcaster extends UDPBroadcaster implements TrackListener {

	public TrackUPDBroadcaster(int port) throws SocketException, UnknownHostException {
		this("localhost", port);
	}

	public TrackUPDBroadcaster(String address, int port) throws SocketException, UnknownHostException {
		super(address, port);
	}

		
	@Override
	public void updateTracks(TrackResults results) {
		this.send(results);		
	}
	
	

}
