package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import net.electroland.eio.InputChannel;
import net.electroland.utils.ParameterMap;

public class TripletCue extends Cue implements ChannelDriven {

    private int timeout, tripInterval;
    private List<String>shows;

    public TripletCue(ParameterMap p) {
        super(p);
        tripInterval = p.getRequiredInt("tripinterval");
        timeout      = p.getRequiredInt("timeout");
        shows        = p.getRequiredList("cues");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        // this method will never be called, since its sensor driven.
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp, InputChannel channel) {
        // only the last channel fired is passed in
        cp.play(shows.get(new Random().nextInt(shows.size())), channel);
    }

    @Override
    public boolean ready(EventMetaData meta) {
        boolean isNotReset    = meta.totalSensorsEvents(tripInterval) > 0;
        boolean isTriple      = meta.totalSensorsEvents(tripInterval * 2) > 2;
        boolean isNotTimedOut = System.currentTimeMillis() - meta.getTimeOfLastCue(this) > timeout;

        return isNotReset && isTriple && isNotTimedOut;
    }
}