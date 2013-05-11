package net.electroland.norfolk.core;

import net.electroland.utils.ElectrolandProperties;

abstract public class Cue {

    public String id;

    public Cue(ElectrolandProperties p){}

    abstract public boolean ready(EventMetaData meta);

    abstract public void fire(EventMetaData meta, ClipPlayer cp);
}