package net.electroland.fish.core;

import javax.vecmath.Vector3f;

/*
 * applied after forces modifys position and/or velocity
 */
public abstract class Constraint implements Comparable<Constraint> {

	boolean isEnabled = true;
	
	protected int priority;
	
	public Constraint(int priority) {
		this.priority = priority;
	}

	public int getPriority() {
		return priority;
	}


	public int compareTo(Constraint o) {
		if(o.getPriority() < priority) return 1;
		if(o.getPriority() == priority) return 0;
		return -1;
	}
	
	public void setEnabled(boolean b) {
		isEnabled = b;
	}
	
	public boolean getEnabled() {
		return isEnabled;
	}

	// use set to modify position and/or velocity
	// return true if there is a change
	abstract public boolean modify(Vector3f position, Vector3f velocity, Boid b) ;

}
