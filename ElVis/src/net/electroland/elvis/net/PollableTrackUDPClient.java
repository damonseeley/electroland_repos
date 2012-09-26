package net.electroland.elvis.net;

import java.net.SocketException;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;

public class PollableTrackUDPClient extends TrackUDPClient {
	protected TrackResults<BaseTrack> mostRecent;
	
	public PollableTrackUDPClient(int port) throws SocketException {
		super(port);
		mostRecent = new TrackResults<BaseTrack>();
	}

	@Override
	public void handle(TrackResults<BaseTrack> t) {
			mostRecent = t;
	}
	
	public TrackResults<BaseTrack> getTracks() {
		return mostRecent;
	}
	

}
