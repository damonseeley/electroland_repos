package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

abstract public class Cue {

    private int time;
    protected Cue parent;
    protected String parentName;
    protected boolean played = false;

    public Cue(ParameterMap params)
    {
        time = params.getRequiredInt("time");
        parentName = params.getOptional(parentName);
    }

    // recursive
    protected int getTime(){
        return parent == null ? time : time + parent.getTime();
    }

    abstract public void play(Map<String, Object> context);
}