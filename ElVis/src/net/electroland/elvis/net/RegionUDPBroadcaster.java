package net.electroland.elvis.net;

import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.Vector;

import net.electroland.elvis.regions.PolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;


public class RegionUDPBroadcaster extends UDPBroadcaster {


	public RegionUDPBroadcaster(int port) throws SocketException, UnknownHostException {
		this("localhost", port);
	}
	public RegionUDPBroadcaster(String address, int port) throws SocketException, UnknownHostException {
		super(address, port);
	}
	
	public void updateRegions(Vector<PolyRegion> regions) {
		if((regions == null) || (regions.isEmpty())) return;
		StringAppender.TrivialAppender results = new StringAppender.TrivialAppender();
		results.setString(PolyRegionResults.buildString(regions));
		this.send(results);
		
	}
	
	

}
