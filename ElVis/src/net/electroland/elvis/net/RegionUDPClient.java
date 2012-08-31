package net.electroland.elvis.net;

import java.net.SocketException;
import java.util.StringTokenizer;

import net.electroland.elvis.regions.BasePolyReagion;
import net.electroland.elvis.regions.PolyRegionResults;

public abstract class  RegionUDPClient extends UDPClient<PolyRegionResults<BasePolyReagion>> {

	public RegionUDPClient(int port) throws SocketException {
		super(port);
	}

	@Override
	public PolyRegionResults<BasePolyReagion> parse(String s) {
		return PolyRegionResults.buildFromStringTokenizer(new StringTokenizer(s, ","));
	}

}
