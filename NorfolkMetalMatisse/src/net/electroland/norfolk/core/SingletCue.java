package net.electroland.norfolk.core;

import net.electroland.eio.InputChannel;
import net.electroland.utils.ParameterMap;

public class SingletCue extends Cue implements ChannelDriven {

    public SingletCue(ParameterMap p) {
        super(p);
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        // this method will never be called, since its sensor driven.
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp, InputChannel channel) {
        cp.play("redRand", channel);
    }

    @Override
    public boolean ready(EventMetaData meta) {
        return true;
    }
}