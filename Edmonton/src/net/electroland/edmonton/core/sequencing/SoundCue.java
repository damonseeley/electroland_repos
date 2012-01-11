package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public class SoundCue extends Cue{

    private int x;
    private int gain;
    private String filename;

    public SoundCue(ParameterMap params)
    {
        super(params);
        filename = params.getRequired("filename");
        x = params.getRequiredInt("x");
        gain = params.getRequiredInt("gain");
    }

    @Override
    public void play(Map<String, Object> context) {
        System.out.println("playing sound " + filename + " at point " + x);
    }
}