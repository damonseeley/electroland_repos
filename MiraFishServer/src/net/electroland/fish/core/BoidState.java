package net.electroland.fish.core;

import java.util.Vector;

import com.electroland.fsm.PollingTransition;
import com.electroland.fsm.State;
import com.electroland.fsm.StateMachine;


public class BoidState extends State {
	Boid boid;
	PollingTransition trans;

	public BoidState(Boid b) {
		boid = b;
	}
	@Override
	public void createTransitions(StateMachine sm) {
		trans = new PollingTransition(sm, this, sm.timer, 500) {
			public State test() {
				return poll();
			}
		};	
	}

	@Override
	public void enterState() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void exitState() {
		// TODO Auto-generated method stub
		
	}
	
	public State poll() {
		return null;
	}
	
	public State onVision(Vector<Boid> boids) {
		return null;
	}
	
	public State onTouch() {
		return null;
	}

}
