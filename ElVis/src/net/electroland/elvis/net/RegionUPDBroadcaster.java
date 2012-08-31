package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Vector;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;


public class RegionUPDBroadcaster extends UDPBroadcaster {


	public RegionUPDBroadcaster(int port) throws SocketException, UnknownHostException {
		this("localhost", port);
	}
	public RegionUPDBroadcaster(String address, int port) throws SocketException, UnknownHostException {
		super(address, port);
	}
	
	public void updateRegions(Vector<PolyRegion> regions) {
		if((regions == null) || (regions.isEmpty())) return;
		StringAppender.TrivialAppender restuls = new StringAppender.TrivialAppender();
		restuls.setString(PolyRegionResults.buildString(regions));
		this.send(restuls);
		
	}
	
	

}
