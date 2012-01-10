package net.electroland.edmonton.core.sequencing;

import java.util.Collection;

import net.electroland.utils.ParameterMap;

public class Piece {

    protected int duration;
    protected String followWith;
    protected Collection<Cue>cues;

    public Piece(ParameterMap params)
    {
        duration = params.getRequiredInt("duration");
        followWith = params.getOptional("follow_with");
    }
    public void reset()
    {
        for (Cue cue: cues)
        {
            cue.played = false;
        }
    }
}