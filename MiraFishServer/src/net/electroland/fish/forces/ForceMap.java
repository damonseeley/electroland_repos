package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class ForceMap extends Force {

	Vector3f[][] map;
	float multiplier;

	public ForceMap(float weight, Boid self, Vector3f[][] map, float multiplier) {
		super(weight, self);
		returnValue.weight = weight;
		this.map = map;
		this.multiplier = multiplier;

	}

	@Override
	public ForceWeightPair getForce() {
		try {
			Vector3f f = map[(int) self.getPosition().x][(int)self.getPosition().y];
			returnValue.force.set(f);			
			returnValue.force.scale(multiplier);
			return returnValue;
		} catch(RuntimeException e) {
			return Force.ZERORETURNVALUE;
		}
	}

}
