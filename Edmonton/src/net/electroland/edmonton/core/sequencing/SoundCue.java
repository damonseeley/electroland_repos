package net.electroland.edmonton.core.sequencing;

import java.util.Map;

import net.electroland.edmonton.core.SoundController;
import net.electroland.utils.OptionException;
import net.electroland.utils.ParameterMap;

import org.apache.log4j.Logger;

public class SoundCue extends Cue{

    static Logger logger = Logger.getLogger(SoundCue.class);

    final static int PLAY_GLOBAL = 0;
    final static int PLAY_SINGLE_CHANNEL = 1;
    final static int PLAY_LOCAL = 2;

    private double x;
    private float gain;
    private String filename;
    private int mode = -1;

    public SoundCue(ParameterMap params)
    {
        super(params);
        String modeStr = params.getRequired("mode");
        if (modeStr.equalsIgnoreCase("PLAY_GLOBAL")){
            mode = SoundCue.PLAY_GLOBAL;
        }else if (modeStr.equalsIgnoreCase("PLAY_SINGLE_CHANNEL"))
        {
            mode = SoundCue.PLAY_SINGLE_CHANNEL;
        }else if (modeStr.equalsIgnoreCase("PLAY_LOCAL"))
        {
            mode = SoundCue.PLAY_LOCAL;
        }else {
            throw new OptionException("unknown mode '" + modeStr + "'");
        }
        filename = params.getRequired("soundfilename");
        x = params.getRequiredDouble("x");
        gain = params.getRequiredDouble("gain").floatValue();
    }

    @Override
    public void play(Map<String, Object> context) {

        if (context != null && context.get("soundController") instanceof SoundController)
        {
            SoundController sc = (SoundController)context.get("soundController");

            switch(mode){
            case(PLAY_GLOBAL):
                sc.playGlobal(filename, false, gain);
                break;
            case(PLAY_LOCAL):
                sc.playLocal(filename, x, gain);
                break;
            case(PLAY_SINGLE_CHANNEL):
                sc.playSingleChannel(filename, x, gain);
                break;
            }
        } else {
            logger.warn("Context is missing 'soundController'");
        }
    }
}