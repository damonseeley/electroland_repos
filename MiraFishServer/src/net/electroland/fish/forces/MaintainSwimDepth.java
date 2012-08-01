package net.electroland.fish.forces;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class MaintainSwimDepth extends Force {
	float min;
	float max;
	float strength;

	public MaintainSwimDepth(float weight, Boid self, float minDepth, float maxDepth, float strength) {
		super(weight, self);
		this.max = maxDepth;
		this.min = minDepth;
		this.strength = strength;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {
		if(self.getPosition().z < min) {
			if(self.getVelocity().z <= 0) {
				returnValue.force.z = strength;
			}
		} else if (self.getProposedPosition().z > max) {
			if(self.getVelocity().z >= 0) {
				returnValue.force.z -= strength;			
			}
		} else {
			return ZERORETURNVALUE;
		}
		return returnValue;
	}

}
