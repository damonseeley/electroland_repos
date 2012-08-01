package com.electroland.fsm;

import java.util.Timer;

public class StateMachine extends Thread {
	public Timer timer = new Timer(); // its going to be used alot so let have it here
	com.electroland.fsm.State curState;
	com.electroland.fsm.State newState;
	boolean isRunning = true;
	
	
	public StateMachine(com.electroland.fsm.State startState) {
		curState = startState;
		start();		
	}
	
	public void exit() {
		isRunning = false;
	}

	public void transition(com.electroland.fsm.State from, com.electroland.fsm.State to) {
		if(from != curState) return; 
		synchronized(this) {
			if(from != curState) return; // check in sync just in case
			if(newState != null) return;
			newState = to;
			notify();
		}
	}

	public void run() {
		while(isRunning) {
			curState.enterState();
			curState.createTransitions(this);
			synchronized(this) {
				if(newState != null) {
					try {
						wait();
					} catch (InterruptedException e) {
						// TODO Auto-generated catch block
						e.printStackTrace();
					}
				}
			}
			curState.cancelTransitions();
			curState.exitState();
			curState = newState;
			newState = null;
		}
	}


}
