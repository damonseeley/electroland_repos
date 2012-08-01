package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class KeepMoving extends Force {
	float strength;

	public KeepMoving(float weight, Boid self, float strength) {
		super(weight, self);
		this.strength = strength;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {
		returnValue.force.set(self.getHeading());
		returnValue.force.scale(strength);
		return returnValue;
		
	}

}
