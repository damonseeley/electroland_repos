package net.electroland.norfolk.core;

import net.electroland.eio.InputChannel;
import net.electroland.utils.ElectrolandProperties;

public class TripletCue extends Cue implements ChannelDriven {

    public TripletCue(ElectrolandProperties p) {
        super(p);
        // TODO Auto-generated constructor stub
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        // TODO Auto-generated method stub

    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp, InputChannel channel) {
        // TODO Auto-generated method stub
    }

    @Override
    public boolean ready(EventMetaData meta) {
        return true;
    }
}