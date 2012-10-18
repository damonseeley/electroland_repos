package net.electroland.gotham.core;

import java.net.SocketException;
import java.util.ArrayList;
import java.util.List;
import java.util.Vector;

import net.electroland.elvis.blobtracking.BaseTrack;
import net.electroland.elvis.blobtracking.TrackResults;
import net.electroland.elvis.net.TrackUDPClient;
import net.electroland.gotham.processing.GothamPApplet;

import org.apache.log4j.Logger;

public class GothamTrackUDPClient extends TrackUDPClient {

    private List <GothamPApplet>listeners;
    static Logger logger = Logger.getLogger(GothamTrackUDPClient.class); 

    public GothamTrackUDPClient(int port) throws SocketException {
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
    public void handle(TrackResults<BaseTrack> t) {

        List<BaseTrack>usableTracks = new ArrayList<BaseTrack>();

        for (BaseTrack tr : t.created) {
            if (!tr.isProvisional) {
                usableTracks.add(tr);
            }
        }

        for (BaseTrack tr : t.existing) {
            if (!tr.isProvisional) {
                usableTracks.add(tr);
            }
        }

        for (GothamPApplet p : listeners){
            p.handle(usableTracks);
        }
    }
}