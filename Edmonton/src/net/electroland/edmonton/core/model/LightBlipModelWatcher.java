package net.electroland.edmonton.core.model;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

public class LightBlipModelWatcher extends ModelWatcher {

    public static int reactivationTimeout;

    private Vector<BlipState> blipState;

    public LightBlipModelWatcher(int reactivationTimeout)
    {
        LightBlipModelWatcher.reactivationTimeout    = reactivationTimeout;
    }


    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        HashMap<String, Object> bmap = new HashMap<String, Object>();

        for (BlipState blip : blipState)
        {
            blip.update();
            LightBlip bp = new LightBlip(blip.state.getLocation().x, blip.renderState == BlipState.ON ? true : false);
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
    public static final int OFF  	= 0;
    public static final int ON 		= 1;
    public static final int WAIT    = 4;
    protected int renderState = OFF;
    private long start;

    protected IState state;

    public BlipState(IState state){
        this.state = state;
    }

    public void update()
    {
        switch(renderState){
            case(OFF): // we can only be turned on if we are currently off.
                if (!state.isSuspect() && state.getState())
                    renderState = ON;
                break;
            case(ON):  // Trigger
                    renderState = WAIT;
                    start = System.currentTimeMillis();
                break;
            case(WAIT): // don't allow another blip until the prescribed waitign period
                if (System.currentTimeMillis() - start > LightBlipModelWatcher.reactivationTimeout){
                    renderState = OFF;
                }
                break;
        }
    }
}