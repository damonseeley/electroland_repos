package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.Bounds;

public class SwimCounterClockwise extends Force {
	Bounds bounds;
	float strength;
	float brakes;

	// rotate clockwise about box defined by bounds
	public SwimCounterClockwise(float weight, Boid self, Bounds bounds, float speed, float brakes) {
		super(weight, self);
		this.bounds = bounds;
		this.strength = speed;
		this.brakes = brakes;
	}

	@Override
	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		returnValue.force = new Vector3f();
		returnValue.weight = 0;

		Bounds.OUTER_REGION region = bounds.getRegion(pos.x, pos.y);
		

		switch(region) {
		case topLeft:
			returnValue.force.y = strength;
			returnValue.weight = weight;
			if(self.getVelocity().x < 0) {
				returnValue.force.x = self.getVelocity().x * - brakes; // slow down if going to wall
			}
			break;
		case left:
			returnValue.force.y = strength;
			returnValue.weight = weight;
			break;
		case bottomLeft:
			returnValue.force.x = strength;
			returnValue.weight = weight;
			if(self.getVelocity().y > 0) {
				returnValue.force.y = self.getVelocity().y * - brakes; // slow down if going to wall
			}
			break;
		case bottom:
			returnValue.force.x = strength;
			returnValue.weight = weight;
			break;
		case bottomRight:
			returnValue.force.y = -strength;
			returnValue.weight = weight;
			if(self.getVelocity().x > 0) {
				returnValue.force.x = self.getVelocity().x * - brakes; // slow down if going to wall
			}
			break;
		case right:
			returnValue.force.y = -strength;
			returnValue.weight = weight;
			break;
		case topRight:
			returnValue.force.x = -strength;
			returnValue.weight = weight;
			
			if(self.getVelocity().y < 0) {
				returnValue.force.y = self.getVelocity().y * - brakes; // slow down if going to wall
			}
			
			break;
		case top:
			returnValue.force.x = -strength;
			returnValue.weight = weight;
			break;


		}
		
		return returnValue;

	}

}
