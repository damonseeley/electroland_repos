package net.electroland.elvis.example;

import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.elvis.net.RegionUDPClient;
import net.electroland.elvis.regions.BasePolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;

public class RegionReceiverExample extends RegionUDPClient {

	public RegionReceiverExample(int port) throws SocketException {
		super(port);
	}


	@Override
	public void handle(PolyRegionResults<BasePolyRegion> t) {
		System.out.println("----");
		for(BasePolyRegion r : t.regions) {
			System.out.println(r.name + " " + r.isTriggered);
		}
		
	}
	
	
	
	public static void main(String args[]) throws SocketException  {
		RegionReceiverExample receiver = new RegionReceiverExample(3457);
		receiver.start();
		
	}

}
