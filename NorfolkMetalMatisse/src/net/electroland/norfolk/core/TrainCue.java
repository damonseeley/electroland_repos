package net.electroland.norfolk.core;

import java.util.List;
import java.util.Random;

import net.electroland.eio.InputChannel;
import net.electroland.utils.ParameterMap;

public class TrainCue extends Cue implements ChannelDriven {

    private List<String> shows;
    private int timeout;

    public TrainCue(ParameterMap p) {
        super(p);
        shows = p.getRequiredList("cues");
        timeout = p.getRequiredInt("timeout");
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        // this method will never be called, since its sensor driven.
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp, InputChannel channel) {
        cp.play(shows.get(new Random().nextInt(shows.size())));
    }

    @Override
    public boolean ready(EventMetaData meta) {
        boolean isNotTimedOut = System.currentTimeMillis() - meta.getTimeOfLastCue(this) > timeout;
        return isNotTimedOut;
    }
}