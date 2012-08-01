package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class RandomTurn extends Force {
	long period;
	long variation;
	float stregth;
	
	long nextTime;
	long appTime;
	

	public RandomTurn(float weight, Boid self, long period, long variation, long appTime, float stregth) {
		super(weight, self);
		this.period = period;
		this.variation = variation;
		this.stregth = stregth;
		nextTime = System.currentTimeMillis() +(long)  ( Math.random() * (period +variation));
		this.appTime = appTime;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {

		if(appTime > self.pond.CUR_TIME) {
			return returnValue;
		} else if(nextTime < self.pond.CUR_TIME) {
			appTime = self.pond.CUR_TIME + appTime;
			nextTime =appTime +  period +  (long) (Math.random() * variation);
			returnValue.force.set(self.getLeft());
			float amount =2f * (float) Math.random() * stregth;
			amount -= stregth;
			returnValue.force.scale(amount);			
			return returnValue;
		} else {
			return ZERORETURNVALUE;
		}
	}

}
