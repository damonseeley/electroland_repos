package net.electroland.elvis.net;

import java.net.SocketException;
import java.util.StringTokenizer;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;

public abstract class TrackUDPClient extends UDPClient<TrackResults<BaseTrack>> {

	public TrackUDPClient(int port) throws SocketException {
		super(port);
		// TODO Auto-generated constructor stub
	}

	@Override
	public TrackResults<BaseTrack> parse(String s) {
		return TrackResults.buildFromString(new StringTokenizer(s, ","));
	}
	

}
