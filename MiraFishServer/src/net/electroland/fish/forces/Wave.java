package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.constraints.FaceVelocity;
import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.constraints.NoEntryMap;
import net.electroland.fish.constraints.WorldBounds;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;

public class Wave extends Force {
	boolean isWaving = false;
	public static enum WaveState { inactive, active }
	public static WaveState waveState = WaveState.inactive;

	float seed;

	double startAng =0;

	public static float waveScalerInc;
//	public static long waveElapsedTime;
//	public static long waveStartTime;



	float speed;
	long rotPeriod;
	double rotScaler;

	public Wave(float weight, Boid self, float speed, long rotPeriod) {
		super(weight, self);
		returnValue.weight = weight;
		this.speed = speed;
		seed = (float)Math.random() + .1f;
		this.rotPeriod = rotPeriod;
		rotScaler = 360.0 /(double)rotPeriod;
	}

	@Override
	public ForceWeightPair getForce() {
		switch(waveState) {
		case inactive:
			if(isWaving) {
				endWave();
			}
			break;
		case active:
			if(isWaving) {
				return getWave();
			} else {
				startWave();
			}
			break;
		}
		return ZERORETURNVALUE;
	}

	public ForceWeightPair getWave() {
		float scale = (waveScalerInc * waveScalerInc) * seed;
		returnValue.force.x += scale;

		Vector3f h = self.getHeading();
		float x = h.x;
		float y = h.y;

		float cos =(float) Math.cos(scale);
		float sin =(float) Math.sin(scale);

		h.x = x * cos - y * sin;
		h.y = x * sin + y * cos;



		return returnValue;
	}

	public void endWave() {
		isWaving = false;

		Constraint c;

		c = self.getConstraint("faceVelocity");
		if(c != null) {
			FaceVelocity fv = (FaceVelocity) c;
			fv.setEnabled(true);
		}

		c = self.getConstraint("noEntry");
		if(c != null) {
			NoEntryMap ne = (NoEntryMap) c;
			ne.setEnabled(true);
		}
		c = self.getConstraint("worldBounds");
		if(c != null) {
			WorldBounds wb = (WorldBounds) c;
			wb.setEnabled(true);
		}
		c = self.getConstraint("maxSpeed");
		if(c != null) {
			MaxSpeed ms = (MaxSpeed) self.getConstraint("maxSpeed");
			ms.resetTempSpeed();
		}

		Force f = self.getForce("dartOnTouch");
		if(f != null) {
			f.setEnabled(true);
		}
		
		if (! self.pond.bounds.contains(self.getPosition().x, self.getPosition().y)) {
			self.offScreen = true;
			self.pond.add(self); // gets added to offscreen list
		}
		

	}

	public void startWave() {

		Constraint c;

		c = self.getConstraint("faceVelocity");
		if(c != null) {
			FaceVelocity fv = (FaceVelocity) c;
			fv.setEnabled(false);
		}

		c = self.getConstraint("noEntry");
		if(c != null) {
			NoEntryMap ne = (NoEntryMap) c;
			ne.setEnabled(false);
		}
		c = self.getConstraint("worldBounds");
		if(c != null) {
			WorldBounds wb = (WorldBounds) c;
			wb.setEnabled(false);
		}

		c = self.getConstraint("maxSpeed");
		if(c != null) {
			MaxSpeed ms = (MaxSpeed) c;
			ms.setTempSpeed(speed);
		}


		Force f = self.getForce("dartOnTouch");
		if(f != null) {
			f.setEnabled(false);
		}

		returnValue.force.set(0,0,0);

		startAng = self.broadcastFish.orientation;

		isWaving = true;

	}

}
