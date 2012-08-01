package net.electroland.fish.forces;

import net.electroland.fish.boids.VideoFish;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class VideoWobble extends Force {
	long period;
	long startTime = -1;
	float amount;
	float baseAngle;
	VideoFish self;
	long halfPeriod;



	public VideoWobble(float weight, Boid self, long period, float amount, float angle, VideoFish vFish) {
		super(weight, self);
		this.period = period;
		this.amount = amount;
		this.baseAngle = angle;
		this.self = vFish;
	}

	boolean countUp = true;
	long oldDiff = -1;
	
	@Override
	public ForceWeightPair getForce() {
		if(startTime <= -1) {
			startTime = self.pond.CUR_TIME;
		}
		
		long diff = self.pond.CUR_TIME - startTime;
		diff %= period;
		
		if(diff <= oldDiff) {
			countUp = ! countUp;			
		}

		oldDiff = diff;
		
		if(!countUp) {
			diff = period - diff;
		} 


		float scale = ((float)diff) / (float) period;	// between 0-1
		scale *= scale; // squared
		scale = 1.0f - scale; // shift the slow change to the edge
		scale -=.5f; // between -.5 and .5
		scale *= 2; // between -1 and 1
		scale *= amount;
		
		self.setAngle(baseAngle + scale);
		
		return ZERORETURNVALUE;
		
	}


	
}
