package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.Bounds;

public class AvoidEdges extends Force {
	Bounds bounds;
	float strength;
	float halfStrength;
	float boost;


	public AvoidEdges(float weight, Boid self, Bounds bounds, float bufferSize, float strength, float boostPercentage) {
		super(weight, self);

		this.bounds = new Bounds(bounds.getTop() + bufferSize,
				bounds.getLeft() + bufferSize,
				bounds.getBottom() - bufferSize,
				bounds.getRight() - bufferSize,
				0,0);

		this.strength = strength;
		halfStrength = .5f * strength;
		boost = boostPercentage;



	}

	@Override
	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		if(! bounds.contains(pos.x, pos.y)) {
			Bounds.OUTER_REGION region = bounds.getRegion(pos.x, pos.y);
			Vector3f dv = new Vector3f();
			switch(region) {

			case topLeft:
				dv.x = halfStrength ;
				dv.y = halfStrength;
				break;
			case left:
				dv.x = strength;
				if(self.getVelocity().x < 0) {
					dv.x -=  self.getVelocity().x * boost;
				}
				break;		
			case bottomLeft:
				dv.x = halfStrength;
				dv.y = -halfStrength;
				break;
			case topRight:
				dv.x = -halfStrength;
				dv.y = halfStrength;
				break;
			case right:
				dv.x = -strength;
				if(self.getVelocity().x > 0) {
					dv.x -=  self.getVelocity().x * boost;
				}
				break;
			case bottomRight:
				dv.x = -halfStrength;
				dv.y = -halfStrength;
				break;		
			case bottom:
				dv.y = -strength;
				if(self.getVelocity().y > 0) {
					dv.y -=  self.getVelocity().y * boost;
				}
				break;
			case top:
				dv.y = strength;
				if(self.getVelocity().y < 0) {
					dv.y -=  self.getVelocity().y * boost;
				}
				break;	
			default:
				returnValue.force = ZERO;
				returnValue.weight = 0;
				return returnValue;


			}

			returnValue.force = dv;
			returnValue.weight = weight;
			return returnValue;


		} else {
			returnValue.force = ZERO;
			returnValue.weight = 0;
			return returnValue;
		}
	}

}
