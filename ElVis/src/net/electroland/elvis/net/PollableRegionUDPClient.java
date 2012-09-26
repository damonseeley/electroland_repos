package net.electroland.elvis.net;

import java.net.SocketException;

import net.electroland.elvis.regions.BasePolyReagion;
import net.electroland.elvis.regions.PolyRegionResults;

public class PollableRegionUDPClient extends RegionUDPClient {
	protected PolyRegionResults<BasePolyReagion> mostRecent;

	public PollableRegionUDPClient(int port) throws SocketException {
		super(port);
		mostRecent = new PolyRegionResults<BasePolyReagion>();
	}

	@Override
	public void handle(PolyRegionResults<BasePolyReagion> t) {
		mostRecent = t;
	}
	
	public PolyRegionResults<BasePolyReagion> getResults() {
		return mostRecent;
	}

}
