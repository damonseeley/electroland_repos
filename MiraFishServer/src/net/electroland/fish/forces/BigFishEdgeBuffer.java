package net.electroland.fish.forces;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class BigFishEdgeBuffer extends Force {
	
	float buffer;
	float strength;

	public BigFishEdgeBuffer(float weight, Boid self, float buffer, float strength) {
		super(weight, self);
		this.buffer = buffer;
		this.strength = strength;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {
		float y = self.getPosition().y;
		float x = self.getPosition().x;

		boolean hasReturn = false;
		returnValue.force.set(0,0,0);
		
		if(y < buffer) {
			returnValue.force.y = strength;
			hasReturn = true;
		} else if (y > self.pond.bounds.getBottom() - buffer) {
			returnValue.force.y = - strength;			
			hasReturn = true;
		}
		
		if(x < buffer) {
			returnValue.force.x = strength;			
			hasReturn = true;
		} else if (x > self.pond.bounds.getRight() - buffer) {
			returnValue.force.x = -strength;			
			hasReturn = true;			
		}
		
		if(hasReturn) {
			return returnValue;
		} else {
			return ZERORETURNVALUE;
		}
		
		
	}

}
