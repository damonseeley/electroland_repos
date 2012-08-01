package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;

public class FaceVelocity extends Constraint {
	float interpSpeed;
	Vector3f heading = new Vector3f();
	Vector3f newHeading = new Vector3f();

	public FaceVelocity(int priority, float interpSpeed) {
		super(priority);
		this.interpSpeed = interpSpeed;
		// TODO Auto-generated constructor stub
	}

	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		newHeading.set(velocity);
		if(newHeading.lengthSquared() > .5f) { // no twitching when stopped
			newHeading.normalize();
			newHeading.scale(interpSpeed);
			heading.scale(1.0f- interpSpeed);
			heading.add(newHeading);
			heading.normalize();
			b.setHeading(heading);
		}
		return false;
	}

}
