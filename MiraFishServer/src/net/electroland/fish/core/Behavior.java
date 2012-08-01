package net.electroland.fish.core;

public abstract class Behavior extends Force {
	// getForce is called once after all objects in view have been seen

	public Behavior(float weight, Boid self) {
		super(weight, self);
	}
	
	abstract public void see(Boid b);
	
}
