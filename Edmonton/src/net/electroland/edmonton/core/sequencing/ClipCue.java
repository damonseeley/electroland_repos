package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

public class ClipCue extends Cue{
    
    final static int PER_SENSOR = 0;
    final static int PER_TRACK = 1;
    final static int GLOBAL = 2;
    
    private double x;
    private int mode = -1;
    private String clipName;

    public ClipCue(ParameterMap params)
    {
        super(params);
        String modeStr = params.getRequired("mode");
        if (modeStr.equalsIgnoreCase("PER_SENSOR")){
            mode = ClipCue.PER_SENSOR;
        }else if (modeStr.equalsIgnoreCase("PER_TRACK"))
        {
            mode = ClipCue.PER_TRACK;
        }else if (modeStr.equalsIgnoreCase("GLOBAL"))
        {
            mode = ClipCue.GLOBAL;
        }else {
            throw new OptionException("unknown mode '" + modeStr + "'");
        }

        clipName = params.getRequired("clip");
        x = params.getRequiredDouble("x");
    }

    @Override
    public void play(Map<String,Object> context) {
        System.out.println("playing clip " + clipName + " at point " + x + " mode=" + mode);
    }
}