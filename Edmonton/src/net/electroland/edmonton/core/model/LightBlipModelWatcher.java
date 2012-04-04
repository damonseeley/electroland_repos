package net.electroland.edmonton.core.model;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

public class LightBlipModelWatcher extends ModelWatcher {

    public static int dbrightness, ddarkness, hold, maxBright, reactivationTimeout;

    private Vector<BlipState> blipState;

    public LightBlipModelWatcher(int dbrightness, int ddarkness, int hold,
                                 int maxBright, int reactivationTimeout)
    {
        LightBlipModelWatcher.dbrightness            = dbrightness;
        LightBlipModelWatcher.ddarkness              = ddarkness;
        LightBlipModelWatcher.hold                   = hold;
        LightBlipModelWatcher.maxBright              = maxBright;
        LightBlipModelWatcher.reactivationTimeout    = reactivationTimeout;
    }


    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        HashMap<String, Object> bmap = new HashMap<String, Object>();

        for (BlipState blip : blipState)
        {
            blip.update();
            BrightPoint bp = new BrightPoint(blip.state.getLocation().x, blip.brightness);
            bp.playSound = blip.playSound;
            bmap.put(blip.state.getID(), bp);
        }
        return bmap;
    }

    @Override
    public boolean poll() {
        if (blipState == null){
            blipState = new Vector<BlipState>();
            for (IState state : this.getStates())
            {
                blipState.add(new BlipState(state));
            }
        }
        return true;
    }
}

class BlipState
{
    private static final int OFF  = 0;
    private static final int GLOW = 1;
    private static final int HOLD = 2;
    private static final int FADE = 3;
    private static final int WAIT = 4;
    private int renderState = OFF;
    private long start;

    protected IState state;
    protected int brightness;
    protected boolean playSound = false;

    public BlipState(IState state){
        this.state = state;
    }

    public void update()
    {
        switch(renderState){
            case(OFF):  // we can only be flipped on if we are already off.
                if (!state.isSuspect() && state.getState())
                    renderState = GLOW;
                    playSound = true;
                break;
            case(GLOW): // ramp up a glow
                brightness += LightBlipModelWatcher.dbrightness;
                playSound = false;
                // when we get to the top, move to the hold state.
                if (brightness > LightBlipModelWatcher.maxBright){
                    brightness = LightBlipModelWatcher.maxBright;
                    renderState = HOLD;
                    start = System.currentTimeMillis();
                }
                break;
            case(HOLD): // hold the brightness until the prescribed delay
                if (System.currentTimeMillis() - start > LightBlipModelWatcher.hold){
                    renderState = FADE;
                }
                break;
            case(FADE): // fade out.
                brightness -= LightBlipModelWatcher.ddarkness;
                // when we bottom out, go into the waiting period
                if (brightness < 0){
                    brightness = 0;
                    renderState = WAIT;
                    start = System.currentTimeMillis();
                }
                break;
            case(WAIT): // don't allow another blip until the prescribed waitign period
                if (System.currentTimeMillis() - start > LightBlipModelWatcher.hold){
                    renderState = OFF;
                }
                break;
        }
    }
}