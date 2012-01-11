package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

abstract public class Cue {

    private int time;
    protected Cue parent;
    protected boolean played = false;
    protected String parentName; // temp buffer
    protected String id; // for debugging only

    public Cue(ParameterMap params)
    {
        time = params.getRequiredInt("time");
        parentName = params.getOptional("parent");
    }

    // recursive
    protected final int getTime(){
        return parent == null ? time : time + parent.getTime();
    }

    abstract public void play(Map<String, Object> context);
}