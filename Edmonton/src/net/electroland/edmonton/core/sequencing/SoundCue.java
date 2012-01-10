package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public class SoundCue extends Cue{

    private int x;
    private String filename;

    public SoundCue(ParameterMap params)
    {
        super(params);
    }

    @Override
    public void play(Map<String, Object> context) {
        System.out.println("playing sound " + filename + " at point " + x);
    }
}