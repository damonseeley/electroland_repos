package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.boids.BigFish1;
import net.electroland.fish.boids.VideoFish;
import net.electroland.fish.boids.BigFish1.State;
import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.core.ContentLayout.Slot;

public class MoveToPointAndPlayVid extends MoveToPoint {

	public Vector3f vidLocation = new Vector3f();
	//float angle;
	float vidHeadingX;
	float vidHeadingY;

	float moveSpeed;
//	boolean cicling = false;
//	long circelTime = 0;

//	SwimInCircle sic;

	VideoFish vid;

//	public MoveToPointAndPlayVid(float weight, Boid self, SwimInCircle sic) {
	public MoveToPointAndPlayVid(float weight, Boid self, float moveSpeed) {
		super(weight, self);
		setEnabled(false);
		this.moveSpeed = moveSpeed;
//		this.sic = sic;
	}

	public void setVideoFish(VideoFish vf) {
		vid = vf;
	}

	@Override
	public ForceWeightPair getForce() {
//		if(cicling) {
		//		if(circelTime > 0) {
		//		circelTime -= self.pond.ELAPSED_TIME;
		//		return sic.getForce();
		//	} else {
		//		cicling= false;

		//			return super.getForce();
//		}
//		} else
		if(atPoint) {
			triggerVid();
			atPoint = false;	
			returnValue.force = ZERO;
			returnValue.weight = 0;
			MaxSpeed ms = (MaxSpeed) self.getConstraint("maxSpeed");
			ms.resetTempSpeed();

			return returnValue;
		} else {
			return super.getForce();
		}
	}

	public void triggerVid() {
		Vector3f pos = new Vector3f(self.getPosition());
		pos.z = 1;
//		System.out.println("playing " + vid.content.name + " at " + self.getPosition() + " and angle" + Boid.vectorToAngle(vidHeadingX, vidHeadingY));
		vid.show(pos, vidHeadingX, vidHeadingY);
		((BigFish1)self).state = State.DIVE;
	}


	public void setEnabled(boolean b) {
		if(b) {
			System.out.println("use other enable");
		}
		else {
			super.setEnabled(b);
		}
	}
	public void enable(boolean b, Slot s) {
		if(b) {
//			sic.setCenter(10);
//			circelTime = 1000;
//			cicling = true;

			Vector3f vidHeading = new Vector3f();
			Vector3f vidLocation = new Vector3f();


			s.getMediaLocation(vid.content, vidLocation, vidHeading);
			vid.slot = s;


			vidHeadingX = vidHeading.x;
			vidHeadingY = vidHeading.y;
			
			MaxSpeed ms = (MaxSpeed) self.getConstraint("maxSpeed");
			ms.setTempSpeed(moveSpeed);

			this.setGoal(vidLocation, 15, moveSpeed);

//			System.out.println("moving to " + vidLocation);



			super.setEnabled(b);


		}
	}
}
