package net.electroland.edmonton.core;

import java.util.Hashtable;

import net.electroland.ea.AnimationManager;
import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.utils.ElectrolandProperties;



public class EIAModel {
	
	private IOManager eio;
	private ElectrolandProperties props;
	private AnimationManager anim;

    private Hashtable <IOState, State> states = new Hashtable<IOState, State>();
    
    public EIAModel(Hashtable context){
    	
		this.eio = (IOManager)context.get("eio");
		this.props = (ElectrolandProperties)context.get("props");
    	this.anim = (AnimationManager)context.get("animmanager");
    	
    	
        for (IOState state : eio.getStates()){
            this.states.put(state, new State((IState)state));
        }
    }
    
    public void update(){
        for (State state : states.values()){
            state.update();
        }
    }
}



// Local class for holding meta logic about sensor states
class State
{
    IState state;

    public State(IState state){
        this.state = state;
    }
    
    public void update()
    {
    	
    }
}