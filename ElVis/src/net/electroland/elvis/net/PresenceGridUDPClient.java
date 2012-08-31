package net.electroland.elvis.net;

import java.net.SocketException;
import java.util.StringTokenizer;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;

public abstract class PresenceGridUDPClient extends UDPClient<GridData> {

	public PresenceGridUDPClient(int port) throws SocketException {
		super(port);
		// TODO Auto-generated constructor stub
	}

	@Override
	public GridData parse(String s) {
		return new GridData(s);
	}
	

}
