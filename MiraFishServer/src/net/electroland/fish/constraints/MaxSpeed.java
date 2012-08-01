package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;

public class MaxSpeed extends Constraint {
	float maxV;
	float maxVSqr;
	
	float defulatMaxSpeed;
	


	public MaxSpeed(int priority, float v) {
		super(priority);
		setSpeed(v);
		defulatMaxSpeed = v;
	}
	
	public void setSpeed(float v) {
		maxV = v;
		maxVSqr = maxV * maxV;
		
	}
	
	public void setTempSpeed(float v) {
		setSpeed(v);
	}
	
	public void resetTempSpeed() {
		setSpeed(defulatMaxSpeed);
	}
	public float getSpeed() {
		return maxV;
	}

	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		float lengthSqr = velocity.lengthSquared();
		if(lengthSqr > maxVSqr) {

			velocity.normalize();
			velocity.scale(maxV);
			
			Vector3f scaledVelocityPerFrame = new Vector3f(velocity);
			scaledVelocityPerFrame.scale(b.pond.CURFRAME_TIME_SCALE);
			position.set(b.getOldPosition());
			position.add(scaledVelocityPerFrame);
			// reset position for new fixed velocity
			
			return true;
		}
		return false;
	}

}
