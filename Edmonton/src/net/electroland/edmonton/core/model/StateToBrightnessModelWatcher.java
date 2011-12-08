package net.electroland.edmonton.core.model;

import java.util.HashMap;
import java.util.Map;
import java.util.Vector;

import net.electroland.eio.IState;
import net.electroland.eio.model.ModelWatcher;

public class StateToBrightnessModelWatcher extends ModelWatcher{

    public static int dbrightness, ddarkness;
    Vector<BrightState> brightStates;

    /**
     * This watcher replicates the behavior of 
     * net.electroland.edmonton.test.TestModel.  You give it an array of
     * IStates, and it watches to see if anyone is in front of any of them.
     * The longer someone is in front of one, the brighter it gets.  After 
     * they leave, it goes dark at a decay rate.
     * 
     * @param dbrightness - how fast it ramps up brightness when someone is in
     * front of it.
     * 
     * @param ddarkness - how fast it ramps down when the person leaves.
     */
    public StateToBrightnessModelWatcher(int dbrightness, int ddarkness)
    {
        StateToBrightnessModelWatcher.dbrightness = dbrightness;
        StateToBrightnessModelWatcher.ddarkness = ddarkness;
        brightStates = new Vector<BrightState>();
        for (IState istate : this.getStates())
        {
            brightStates.add(new BrightState(istate));
        }
    }

    /**
     * This watcher ALWAYS returns an event.  The event contains a Map of the
     * IState values translated to a level of brightness.  This is watcher
     * reproduces the functionality of the original 
     * net.electroland.edmonton.test.TestModel class.
     */
    @Override
    public boolean poll() {
        // as noted: ALWAYS return an event.
        return true;
    }

    /**
     * In the event, retuns a map of the ID of each IState and it's current 
     * level of "brightness."  Brightness is just a factor of how long someone
     * has been in front of a sensor.  There's a ramp up, and a ramp down. 
     */
    @Override
    public Map<String, Object> getOptionalPositiveDetails() {
        HashMap<String, Object> bmap = new HashMap<String, Object>();
        for (BrightState bright : brightStates)
        {
            bright.update();
            bmap.put(bright.state.getID(), bright.brightness);
        }
        return bmap;
    }
}

class BrightState
{
    IState state;
    int brightness;
 
    public BrightState(IState state){
        this.state = state;
    }
    public void update()
    {
        if (state.getState()){
            // fade on
            brightness += StateToBrightnessModelWatcher.dbrightness;
            if (brightness > 255){
                brightness = 255;
            }
        }else{
            // fade off
            brightness -= StateToBrightnessModelWatcher.ddarkness;
            if (brightness < 0){
                brightness = 0;
            }
        }
    }
}