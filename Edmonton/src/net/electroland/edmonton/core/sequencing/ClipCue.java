package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public class ClipCue extends Cue{

    private int x;
    private boolean isPerSensor;
    private String clipName;

    public ClipCue(ParameterMap params)
    {
        super(params);
        isPerSensor = params.getRequired("mode").equalsIgnoreCase("PER_SENSOR");
        clipName = params.getRequired("clip");
    }

    @Override
    public void play(Map<String,Object> context) {
        System.out.println("playing clip " + clipName + " at point " + x + " isPerSensor=" + isPerSensor);
    }
}