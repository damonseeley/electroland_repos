package net.electroland.fish.forces;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.ForceWeightPair;

public class FollowPath extends MoveToPoint {

	public FollowPath(float weight, Boid self) {
		super(weight, self);
	}

	@Override
	public ForceWeightPair getForce() {
		ForceWeightPair v = super.getForce();
		
		
		
		return v;
	}
}
