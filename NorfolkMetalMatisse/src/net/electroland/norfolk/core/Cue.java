package net.electroland.norfolk.core;

abstract public class Cue {

    public String id;

    abstract public void fire(EventMetaData meta, ClipPlayer cp);
}