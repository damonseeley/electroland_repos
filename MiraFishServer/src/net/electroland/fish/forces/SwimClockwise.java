package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.Bounds;

public class SwimClockwise extends Force {
	Bounds bounds;
	float strength;

	// rotate clockwise about box defined by bounds
	public SwimClockwise(float weight, Boid self, Bounds bounds, float strength) {
		super(weight, self);
		this.bounds = bounds;
		this.strength = strength;
	}

	@Override
	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		returnValue.force = new Vector3f();
		returnValue.weight = 0;
		if(pos.x < bounds.getLeft()) {
			if(pos.y < bounds.getTop()) {
				returnValue.force.x = strength;
				returnValue.weight = weight;
				if(self.getVelocity().y < 0) {
					returnValue.force.y = self.getVelocity().y * - .001f; // slow down if going to wall
				}
			} else if (pos.y > bounds.getBottom()) {
				if(self.getVelocity().x < 0) {
					returnValue.force.x = self.getVelocity().x * - .001f; // slow down if going to wall
				}
				returnValue.force.y = -strength;
				returnValue.weight = weight;
			} 
		} else if (pos.x > bounds.getRight()) {
			if(pos.y < bounds.getTop()) {
				if(self.getVelocity().x > 0) {
					returnValue.force.x = self.getVelocity().x * - .001f; // slow down if going to wall
				}
				returnValue.force.y = strength;
				returnValue.weight = weight;
			} else if (pos.y > bounds.getBottom()) {
				returnValue.force.x = -strength;
				returnValue.weight = weight;
				if(self.getVelocity().y > 0) {
					returnValue.force.y = self.getVelocity().y * - .001f; // slow down if going to wall
				}
			}			
		}
		return returnValue;
		
		
		
	}

}
