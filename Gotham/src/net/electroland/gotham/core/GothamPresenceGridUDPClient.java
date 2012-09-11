package net.electroland.gotham.core;

import java.net.SocketException;
import java.util.List;
import java.util.Vector;

import net.electroland.elvis.net.GridData;
import net.electroland.elvis.net.PresenceGridUDPClient;
import net.electroland.gotham.processing.GothamPApplet;

public class GothamPresenceGridUDPClient extends PresenceGridUDPClient {

    private List <GothamPApplet>listeners;

    public GothamPresenceGridUDPClient(int port) throws SocketException {
        super(port);
        listeners = new Vector<GothamPApplet>();
    }

    @Override
    public void handel(GridData t) {
        for (GothamPApplet p : listeners){
            p.handle(t);
        }
    }

    public void addListener(GothamPApplet listener){
        listeners.add(listener);
    }

    public void removeListener(GothamPApplet listener){
        listeners.remove(listener);
    }
}