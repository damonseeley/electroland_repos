package com.electroland.fsm;

import java.util.Timer;
import java.util.TimerTask;

public abstract class PollingTransition extends Transition {
	TimerTask task;
	

	public PollingTransition(StateMachine stateMachine, State state, Timer timer, long pollingFreqency) {
		super(stateMachine, state);
		task = new Poller();
		timer.scheduleAtFixedRate(task, 1, pollingFreqency); // do furst check asap
	}

	public void cancel() {
		task.cancel();
	}
	
	/*
	 * returns new state if transition is needed else null
	 */
	public abstract State test(); 
	
	public class Poller extends TimerTask {

		public void run() {
			State nextState = test();
			if(nextState != null) {
				stateMachine.transition(state, nextState);
//				stateMachine.transition(state, nextState);
			}
		}
		
	}
	

}
