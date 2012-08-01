package net.electroland.fish.behaviors;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Behavior;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.ForceWeightPair;

public class AvoidSameSpecies extends Behavior {
	boolean isAvoiding = false;
	float desiredSeparation;

	Vector3f avoidenceVector = new Vector3f();	
	Vector3f result = new Vector3f();


	Vector3f run = new Vector3f();

	public long runTime =0;

	float scalar;

	public AvoidSameSpecies(float weight, Boid self, float desiredSeparation, float scalar) {
		super(weight, self);
		this.desiredSeparation = desiredSeparation;
		this.scalar = scalar;
	}


	public void avoid(Boid b) {
		avoidenceVector.set(self.getPosition());
		avoidenceVector.sub(b.getPosition());
		float dist = avoidenceVector.length();

		if(dist <= 6) { // we don't want nands after we normalize
//			System.out.println("stacked");
//			Vector3f v = new Vector3f(b.getHeading());

			if(runTime < self.pond.CUR_TIME) {
				run.set((float)Math.random()-.5f, (float)Math.random()-.5f,0);
				run.scale(1);
			}


			runTime = self.pond.CUR_TIME + 2000;
			result.add(run);
			isAvoiding = true;
		} else {
			if(runTime > self.pond.CUR_TIME) {
				result.add(run);
				isAvoiding = true;
			} else {
				float separation = dist;
				dist -= self.getSize();
				dist -= b.getSize();


				if(dist < desiredSeparation) {
//					System.out.println("     " + dist + "<" + desiredSeparation);
					avoidenceVector.scale(scalar / separation); // normalize
					result.add(avoidenceVector);
					isAvoiding = true;
				}
			}

		}
	}

	@Override
	public void see(Boid b) {

		if(b == self) return;

		if((b.getFlockId() == self.getFlockId()) && (b.subFlockId == b.subFlockId)){
			avoid(b);
		}

	}

	@Override
	public ForceWeightPair getForce() {

		returnValue.force.set(result);
		result.set(0,0,0);

		if(runTime > self.pond.CUR_TIME) {
			returnValue.force.set(run);
			returnValue.weight = weight * 2f;
			isAvoiding = false;
		} else if(isAvoiding) {
			returnValue.weight = weight;
			isAvoiding = false;
		} else {
			returnValue.weight = 0;
		}
		return returnValue;		
	}


}
