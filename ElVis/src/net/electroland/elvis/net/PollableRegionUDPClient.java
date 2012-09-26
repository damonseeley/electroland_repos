package net.electroland.elvis.net;

import java.net.SocketException;

import net.electroland.elvis.regions.BasePolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;

public class PollableRegionUDPClient extends RegionUDPClient {
	protected PolyRegionResults<BasePolyRegion> mostRecent;

	public PollableRegionUDPClient(int port) throws SocketException {
		super(port);
		mostRecent = new PolyRegionResults<BasePolyRegion>();
	}

	@Override
	public void handle(PolyRegionResults<BasePolyRegion> t) {
		mostRecent = t;
	}
	
	public PolyRegionResults<BasePolyRegion> getResults() {
		return mostRecent;
	}

}
