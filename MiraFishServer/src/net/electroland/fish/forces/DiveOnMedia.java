package net.electroland.fish.forces;

import net.electroland.fish.boids.BigFish1;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.core.ContentLayout.Slot;

public class DiveOnMedia extends Force {

	BigFish1 self;

	float speed;

	public DiveOnMedia(float weight, BigFish1 self, float speed) {
		super(weight, self);
		this.speed = speed;
		returnValue.weight = weight;
		this.self = self;
		super.self = self;
	}

	@Override
	public ForceWeightPair getForce() {
//		if(self.getPosition().z <= 1) {
//			self.broadcastFish.scale = self.getPosition().z;
//		} else {
//			self.broadcastFish.scale = 1;
//		}
		
		if((self.state == BigFish1.State.SWIMING) || self.state == BigFish1.State.DIVE) {
			Slot s = self.pond.contentLayout.getFreeSlot(self.getPosition());

			if(s == null) {
				self.state = BigFish1.State.DIVE;
				if(self.getPosition().z > .25) {
					returnValue.force.z = -speed;
					return returnValue;
				} else {
					return ZERORETURNVALUE;
				}
			} else {
				self.state = BigFish1.State.SWIMING;
				if(self.getPosition().z < 1.5) {
					returnValue.force.z = speed;
					return returnValue;
				} else {
					return ZERORETURNVALUE;
				}
			}
		} else {
			if(self.getPosition().z < 1.5) {
				returnValue.force.z = speed;
				return returnValue;
			} else {
				return ZERORETURNVALUE;
			}
	
		}
	}

}
