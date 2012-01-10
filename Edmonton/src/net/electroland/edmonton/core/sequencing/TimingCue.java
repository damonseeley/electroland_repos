package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.ParameterMap;

public class TimingCue extends Cue {

    public TimingCue(ParameterMap params)
    {
        super(params);
        // no extra params
    }

    @Override
    public void play(Map<String, Object> context) {
        // do nothing.
    }
}