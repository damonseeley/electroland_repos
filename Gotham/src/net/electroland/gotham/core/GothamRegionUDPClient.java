package net.electroland.gotham.core;

import java.net.SocketException;
import java.util.List;
import java.util.Vector;

import org.apache.log4j.Logger;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;
import net.electroland.elvis.net.RegionUDPClient;
import net.electroland.elvis.regions.BasePolyRegion;
import net.electroland.elvis.regions.PolyRegionResults;
import net.electroland.gotham.processing.GothamPApplet;

public class GothamRegionUDPClient extends RegionUDPClient {

    private List <GothamPApplet>listeners;
    static Logger logger = Logger.getLogger(GothamRegionUDPClient.class); 

    public GothamRegionUDPClient(int port) throws SocketException {
        super(port);
        listeners = new Vector<GothamPApplet>();
        logger.info("Region Detector Client started");
    }

    public void addListener(GothamPApplet listener){
        listeners.add(listener);
    }

    public void removeListener(GothamPApplet listener){
        listeners.remove(listener);
    }

	@Override
	public void handle(PolyRegionResults<BasePolyRegion> t) {
		// TODO Auto-generated method stub
		logger.info("Region: " + t);
		
	}
}