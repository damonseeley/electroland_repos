package net.electroland.fish.behaviors;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Behavior;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.ForceWeightPair;

public class MatchFlockVelocity extends Behavior {
	int cnt = 0;
	Vector3f perceivedVelocity= new Vector3f();
	Vector3f result = new Vector3f();


	public MatchFlockVelocity(float weight, Boid self) {
		super(weight, self);
		
	}

	@Override
	public void see(Boid b) {
		if((self.getFlockId() == b.getFlockId() && (self.subFlockId == b.subFlockId))) {
			cnt++;
			perceivedVelocity.add(b.getVelocity());			
		}
	}

	@Override
	public ForceWeightPair getForce() {
		if(cnt > 1) { // if 1 only has seen self
			perceivedVelocity.scale(1.0f/(float) cnt);
			returnValue.force.set(perceivedVelocity);			
			returnValue.force.sub(self.getVelocity());
			returnValue.weight = weight;
		} else {
			returnValue.force.set(0,0,0);
			returnValue.weight = 0;
		}

		perceivedVelocity.set(0,0,0);
		cnt = 0;
		
		return returnValue;
	}

}
