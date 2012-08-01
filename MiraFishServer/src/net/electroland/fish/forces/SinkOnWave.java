package net.electroland.fish.forces;

import net.electroland.fish.boids.VideoFish;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class SinkOnWave extends Force {
	long sinkTime;
	long sinkStartTime;
	float startZ;
	float startScale;

	public SinkOnWave(float weight, Boid self, long time) {
		super(weight, self);
		sinkTime = time;
		sinkStartTime = -1;
		
	}

	@Override
	public ForceWeightPair getForce() {
		if(Wave.waveState == Wave.WaveState.active) {
			if(sinkStartTime < 0) {
				sinkStartTime = self.pond.CUR_TIME;
				startZ = self.getPosition().z;
				startScale = (float) self.broadcastFish.scale;
			} else {
				long diff = self.pond.CUR_TIME - sinkStartTime;
				float scale = 1.0f - ((float) diff) / sinkTime;
				if(scale < 0) {
					self.getPosition().z = 0;
					((VideoFish) self).killTime = 0;
				} else {
					self.getPosition().z = startZ * scale;
					self.broadcastFish.scale = scale;
				}
			}
		}
		return ZERORETURNVALUE;
	}

}
