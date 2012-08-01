package com.electroland.fsm;

import java.util.Vector;

public abstract class State {
	boolean transitioned = false;
	
	Vector<Transition> transitions = new Vector<Transition>();
	
	
	public abstract void enterState();
	public abstract void exitState();
	public abstract void createTransitions(StateMachine stateMachine);
	

	
	public void addTransition(Transition transition) {
		if (transitioned) return;
		synchronized(transitions) {
			if (transitioned) return;
			transitions.add(transition);			
		}
	}
	
	public void cancelTransitions() {
		transitioned = true;
		synchronized(transitions) {
			for(Transition t : transitions) {
				t.cancel();
			}
		}
		
	}

}
