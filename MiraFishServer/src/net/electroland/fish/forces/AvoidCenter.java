package net.electroland.fish.forces;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.Bounds;

public class AvoidCenter extends Force {
	Bounds islandBounds;
	Bounds outerBounds;
	float strength;
	float halfStrength;


	public AvoidCenter(float weight, Boid self, Bounds islandBounds, float bufferSize, float strength) {
		super(weight, self);

		this.islandBounds = islandBounds;
		outerBounds = new Bounds(islandBounds.getTop() - bufferSize,
				islandBounds.getLeft() - bufferSize,
				islandBounds.getBottom() + bufferSize,
				islandBounds.getRight() + bufferSize,
				0,0);

		this.strength = strength;
		halfStrength = .5f * strength;



	}

	@Override
	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		if(outerBounds.contains(pos.x, pos.y)) {
			Bounds.OUTER_REGION region = islandBounds.getRegion(pos.x, pos.y);
			Vector3f dv = new Vector3f();
			switch(region) {

			case topLeft:
				dv.x = -halfStrength ;
				dv.y = -halfStrength;
				break;
			case left:
				dv.x = -strength;
				break;		
			case bottomLeft:
				dv.x = -halfStrength;
				dv.y = halfStrength;
				break;
			case topRight:
				dv.x = halfStrength;
				dv.y = -halfStrength;
				break;
			case right:
				dv.x = strength;
				break;		
			case bottomRight:
				dv.x = halfStrength;
				dv.y = halfStrength;
				break;		
			case bottom:
				dv.y = strength;
				break;
			case top:
				dv.y = -strength;
				break;	
			default:
				returnValue.force = ZERO;
				returnValue.weight = 0;
				return returnValue;


			}

			returnValue.force = dv;
			returnValue.weight = weight;
			return returnValue;


		} 
		returnValue.force = ZERO;
		returnValue.weight = 0;
		return returnValue;
	}

}
