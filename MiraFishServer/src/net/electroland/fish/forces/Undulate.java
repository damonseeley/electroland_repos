package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.Bounds;

public class Undulate extends Force {

	Bounds innerBounds;
	float strength;
	float halfStrength;
	long period;
	long timeLeft;

	/**
	 * period in ms
	 */	
	public Undulate(float weight, Boid self, Bounds innerBounds, float strength, long period) {
		super(weight, self);

		this.innerBounds =innerBounds;
		this.strength = strength;
		this.halfStrength = strength * .5f;  // weaker in corners for turn
		this.period = period;
		this.timeLeft = period;
	}

	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		returnValue.force = new Vector3f();

		timeLeft -= self.pond.ELAPSED_TIME;
		if(timeLeft <= 0) {
			timeLeft = period;
			strength *= -1.0f;
			halfStrength *= -1.0f;
			
		}

		Bounds.OUTER_REGION region = innerBounds.getRegion(pos.x, pos.y);


		switch(region) {

		case topLeft:
			returnValue.force.x = halfStrength ;
			returnValue.force.y = halfStrength;
			break;
		case left:
			returnValue.force.x = strength;
			break;		
		case bottomLeft:
			returnValue.force.x = halfStrength;
			returnValue.force.y = -halfStrength;
			break;
		case topRight:
			returnValue.force.x = -halfStrength;
			returnValue.force.y = halfStrength;
			break;
		case right:
			returnValue.force.x = -strength;
			break;		
		case bottomRight:
			returnValue.force.x = -halfStrength;
			returnValue.force.y = -halfStrength;
			break;		
		case bottom:
			returnValue.force.y = -strength;
			break;
		case top:
			returnValue.force.y = strength;
			break;	


		}

		return returnValue;


	}

}
