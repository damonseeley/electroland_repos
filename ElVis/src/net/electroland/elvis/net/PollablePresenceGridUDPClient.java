package net.electroland.elvis.net;

import java.net.SocketException;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;

public class PollablePresenceGridUDPClient extends PresenceGridUDPClient {
	
	protected GridData mostRecent;
	
	public PollablePresenceGridUDPClient(int port) throws SocketException {
		super(port);
	}

	@Override
	public void handel(GridData t) {
			mostRecent = t;
	}
	
	public GridData getTracks() {
		return mostRecent;
	}
	

}
