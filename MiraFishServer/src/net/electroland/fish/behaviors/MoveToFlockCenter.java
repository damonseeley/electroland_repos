package net.electroland.fish.behaviors;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Behavior;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.ForceWeightPair;

public class MoveToFlockCenter extends Behavior {
	int cnt = 0;
	Vector3f perceivedCenter = new Vector3f();
	Vector3f result = new Vector3f();

	Vector3f tmp = new Vector3f();
	
	float strength;

	public MoveToFlockCenter(float weight, Boid self, float strength) {
		super(weight, self);
		this.strength = strength;
	}


	@Override
	public void see(Boid b) {
		if((self.getFlockId() == b.getFlockId() && (self.subFlockId == b.subFlockId))) {

			tmp.set(b.getPosition());
			tmp.sub(self.getPosition());
			if(tmp.lengthSquared() > 16) {	// ignore if too close don't want to collide		
				perceivedCenter.add(b.getPosition());
				cnt++;
			}
		}
	}

	@Override
	public ForceWeightPair getForce() {
		if(cnt > 1) { // if 1 only has seen self
			perceivedCenter.scale(1.0f/(float) cnt);

			returnValue.force.set(perceivedCenter);
			returnValue.force.sub(self.getPosition());
			returnValue.force.scale(strength);
			returnValue.weight = weight;
		} else {
			returnValue.force.set(0,0,0);
			returnValue.weight = 0;
		}

		perceivedCenter.set(0,0,0);
		cnt = 0;


		return returnValue;

	}


}
