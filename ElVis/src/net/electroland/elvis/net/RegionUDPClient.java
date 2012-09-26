package net.electroland.elvis.net;

import java.net.SocketException;
import java.util.StringTokenizer;

import net.electroland.elvis.regions.BasePolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;

public abstract class  RegionUDPClient extends UDPClient<PolyRegionResults<BasePolyRegion>> {

	public RegionUDPClient(int port) throws SocketException {
		super(port);
	}

	@Override
	public PolyRegionResults<BasePolyRegion> parse(String s) {
		return PolyRegionResults.buildFromStringTokenizer(new StringTokenizer(s, ","));
	}

}
