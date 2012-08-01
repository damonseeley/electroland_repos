package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.constraints.MaxSpeed;
import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.FishProps;

public class DartOnTouch extends Force {

	long dartUntilTime;

	long dartTime = 1000;
	float speed;


	MaxSpeed maxSpeed;
	boolean reset = true;

	Vector3f dir = new Vector3f();;


	public static final float dartSafteyBuffer = FishProps.THE_FISH_PROPS.getProperty("dartSafteyBuffer", 100);





	public DartOnTouch(float weight, Boid self, float speed, int time) {
		super(weight, self);
		this.speed = speed;
		this.dartTime = time;
		Constraint c = self.getConstraint("maxSpeed");
		if(c != null) {
			maxSpeed = (MaxSpeed)c;
		}

		dir.set(1 - (float)Math.random() *2f,1 - (float)Math.random()*2f,0 );
		dir.normalize();
		dir.scale(speed);
		returnValue.force = dir;
		returnValue.weight = weight;



	}


	public void setDir() {
		dir.set(1 - (float)Math.random() *2f,1 - (float)Math.random()*2f,0 );
		dir.normalize();
		dir.scale(speed);

		if(self.getPosition().x < dartSafteyBuffer) {
			dir.x = (dir.x < 0) ? - dir.x : dir.x;

		} else if (self.getPosition().x > self.pond.bounds.getRight() - dartSafteyBuffer) {
			dir.x = (dir.x > 0) ? - dir.x : dir.x;
		}
		
		if(self.getPosition().y < dartSafteyBuffer) {
			dir.y = (dir.y < 0) ? - dir.y : dir.y;

		} else if (self.getPosition().y > self.pond.bounds.getBottom() - dartSafteyBuffer) {
			dir.y = (dir.y > 0) ? - dir.y : dir.y;
		}

		returnValue.force = dir;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {

		if(dartUntilTime > self.pond.CUR_TIME) {
			return returnValue;
		} else if(self.isTouched) {
			setDir();

			if(maxSpeed != null) {
				float curSpeed = maxSpeed.getSpeed();
				if(curSpeed < speed) {
					maxSpeed.setTempSpeed(speed);					
				}
				reset = true;
			}
			dartUntilTime = self.pond.CUR_TIME + dartTime + (long) (Math.random() * 200.0);
			return returnValue;
		}else {
			if(reset) {
				maxSpeed.resetTempSpeed();
				reset = false;				
			}			
			return ZERORETURNVALUE;
		}

	}

}
