package net.electroland.fish.core;

import javax.vecmath.Vector3f;

/*
 * exerts a change in velocity 
 */
public abstract class Force  {
	protected Boid self;
	
	protected float weight;
	
	public static final Vector3f ZERO = new Vector3f(0,0,0);
	public static final ForceWeightPair ZERORETURNVALUE = new ForceWeightPair(ZERO,0);
	
	protected ForceWeightPair returnValue = new ForceWeightPair(); // using this is encoraged (as opposed to creating object every frame)

	
	private boolean isEnabled = true;
	
	public Force(float weight, Boid self) {
		this.weight = weight;
		this.self = self;
		returnValue.force = new Vector3f() ;
	}
	
	
	public float getWeight() {
		return weight;		
	}

	public void setWeight(float w) {
		weight = w;
	}
	
//	public Vector3f getDeltaV() {
//		force.set(getForce());		
//		force.scale(weight);
//		return force;	
//	}
	
	abstract public ForceWeightPair getForce();


	public boolean isEnabled() {
		return isEnabled;
	}


	public void setEnabled(boolean isEnabled) {
		this.isEnabled = isEnabled;
	}
	


	
}
