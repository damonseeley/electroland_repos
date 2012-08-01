package com.electroland.fsm;

public abstract class Transition {
	StateMachine stateMachine;
	State state;
	
	public Transition(StateMachine stateMachine, State state) {
		this.stateMachine = stateMachine;
		this.state = state;
		state.addTransition(this);
	}
	
	
	public void transition(State newState) {
		stateMachine.transition(state, newState);
	}
	
	public abstract void cancel();
	
	

}
