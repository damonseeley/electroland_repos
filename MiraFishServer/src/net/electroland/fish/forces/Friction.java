package net.electroland.fish.forces;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class Friction extends Force {

	float percent;
	
	public Friction(float weight, Boid self, float percent) {
		super(weight, self);
		this.percent = percent;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {
		returnValue.force.set(self.getVelocity());
		returnValue.force.scale(-percent);
		return returnValue;
		
	}

}
