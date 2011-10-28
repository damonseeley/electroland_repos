package net.electroland.edmonton.test;

import java.util.Collection;
import java.util.Hashtable;

import net.electroland.eio.IOState;
import net.electroland.eio.IState;

public class TestModel {

    Hashtable <IOState, State> states = new Hashtable<IOState, State>();

    public TestModel(Collection <IOState> states, int dbrightness){
        for (IOState state : states){
            this.states.put(state, new State((IState)state, dbrightness));
        }
    }
    public void update(){
        for (State state : states.values()){
            state.update();
        }
    }
    public int getBrightness(IOState state){
        return states.get(state).brightness;
    }
}
class State
{
    IState state;
    int dbrightness;
    int brightness;

    public State(IState state, int dbrightness){
        this.dbrightness = dbrightness;
        this.state = state;
    }
    public void update()
    {
        if (state.getState()){
            // fade on
            brightness += dbrightness;
            if (brightness > 255){
                brightness = 255;
            }
        }else{
            // fade off
            brightness -= dbrightness;
            if (brightness < 0){
                brightness = 0;
            }
        }
    }
}