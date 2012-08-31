package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;

import com.googlecode.javacv.cpp.opencv_core.IplImage;

public class PresenceGridUPDBroadcaster extends UDPBroadcaster {

	public PresenceGridUPDBroadcaster(int port) throws SocketException,
	UnknownHostException {
		super("localhost", port);
	}
	public PresenceGridUPDBroadcaster(String address, int port) throws SocketException,
	UnknownHostException {
		super(address, port);
	}

	
	public void updateGrid(IplImage img) {
		if(img == null) return;
		send(new GridData(img));
	}
}
