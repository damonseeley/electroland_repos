package net.electroland.norfolk.core;

import net.electroland.utils.ElectrolandProperties;

public class TimedCue extends Cue {

    public TimedCue(ElectrolandProperties p) {
        super(p);
        // TODO Auto-generated constructor stub
    }

    @Override
    public void fire(EventMetaData meta, ClipPlayer cp) {
        // TODO Auto-generated method stub

    }

    @Override
    public boolean ready(EventMetaData meta) {
        // TODO Auto-generated method stub
        return false;
    }
}