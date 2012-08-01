package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class MoveToPoint extends Force {
	boolean atPoint = false;
	Vector3f goal;
	Vector3f tmp;
	float radiusSqr;
	float speed;

	public MoveToPoint(float weight, Boid self) {
		super(weight, self);
	}

	public void setGoal(Vector3f vec, float radius, float speed) {
		goal = vec;
		this.speed = speed;
		tmp = new Vector3f();
		radiusSqr = radius * radius;
		atPoint = false;
		returnValue.force.set(0,0,0);
		returnValue.weight = 0;

	}

	

	
	@Override
	public ForceWeightPair getForce() {
		
		if(goal == null) {
			return returnValue; // should be set to 0 already
		}


		tmp.sub(goal, self.getPosition());



		
		if(tmp.lengthSquared() < radiusSqr) {
			goal = null;
			atPoint = true;

			return returnValue;
		}
		


		tmp.normalize();

		tmp.scale(speed);
		
		// dampen momentum
		Vector3f v = new Vector3f(self.getVelocity());
		v.scale(-.1f);
		tmp.add(v);
	
	
		returnValue.force = tmp;
		returnValue.weight = weight;
		return returnValue;		
	}

}
