package net.electroland.gotham.core;

import java.net.SocketException;
import java.util.List;
import java.util.Vector;

import org.apache.log4j.Logger;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;
import net.electroland.gotham.processing.GothamPApplet;

public class GothamPresenceGridUDPClient extends PresenceGridUDPClient {

    private List <GothamPApplet>listeners;
    static Logger logger = Logger.getLogger(GothamPresenceGridUDPClient.class); 

    public GothamPresenceGridUDPClient(int port) throws SocketException {
        super(port);
        listeners = new Vector<GothamPApplet>();
    }

    @Override
    public void handle(GridData t) {
    	//logger.info("GridData: " + containsPresence(t));
        for (GothamPApplet p : listeners){
            p.handle(t);
        }

    }

    public boolean containsPresence(GridData t){
    	for (int i = 0 ; i < t.data.length; i++){
    		if (t.data[i] != (byte)0) return true;
    	}
    	return false;
    }
    
    public void addListener(GothamPApplet listener){
        listeners.add(listener);
    }

    public void removeListener(GothamPApplet listener){
        listeners.remove(listener);
    }
}