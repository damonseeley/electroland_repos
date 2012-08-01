package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.core.Vector2fDirPreserver;

public class SwimInCircle extends Force {
//	MoveToPoint mtp;
	float strength;
	float radius;
	Vector3f center;



	public SwimInCircle(float weight, Boid self, float strength) {
		super(weight, self);
		this.strength = strength;

	}
	public void setCenter(Vector3f center) {
		this.center = center;
	}

	public void setCenter(float r) {
		Vector3f vec = new Vector3f(self.getVelocity());
		Vector2fDirPreserver v = new Vector2fDirPreserver(vec.x, vec.y);
		v.normalize();
		v.perp();
		v.scale(r);
		v.x += self.getPosition().x;
		v.y += self.getPosition().y;		
		vec.x = v.x;
		vec.y = v.y;
		setCenter(vec);
	}

	@Override
	public ForceWeightPair getForce() {
		returnValue.weight = weight;
		returnValue.force= new Vector3f();
		returnValue.force.sub(self.getPosition(), center);
		returnValue.force.z = 0f;
		returnValue.force.normalize();
		float x = returnValue.force.y;
		returnValue.force.y = -returnValue.force.x;
		returnValue.force.x = x;			
		returnValue.force.scale(-strength);
		return returnValue;


	}

}
