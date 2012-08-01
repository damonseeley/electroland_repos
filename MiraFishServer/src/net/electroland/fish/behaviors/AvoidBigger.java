package net.electroland.fish.behaviors;

import net.electroland.fish.core.Boid;

public class AvoidBigger extends AvoidSameSpecies {

	public AvoidBigger(float weight, Boid self, float desiredSeparation,
			float scalar) {
		super(weight, self, desiredSeparation, scalar);
	}

	
	@Override
	public void see(Boid b) {

		if(b == self) return;

		if(b.getFlockId() != self.getFlockId()) {
			if(b.getSize() >= self.getSize()) {
				avoid(b);
			}
		}

	}
}
