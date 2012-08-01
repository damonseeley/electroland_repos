package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;
import net.electroland.fish.util.Bounds;

public class WorldBounds extends Constraint {
	Bounds bounds;

	public WorldBounds(Bounds bounds) {
		super(Integer.MAX_VALUE); // make sure applied last
		this.bounds = bounds;
	}
	
	
	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		boolean returnVal = false;
		
		if(b.offScreen) {
			return false;
		} 
		
		
		if(position.x < bounds.getLeft()) {
			position.x = bounds.getLeft();
			returnVal = returnVal || true;
		}
		
		if(position.x > bounds.getRight()) {
			position.x = bounds.getRight();
			returnVal = returnVal || true;
		}

		if(position.y < bounds.getTop()) {
			position.y = bounds.getTop();
			returnVal = returnVal || true;
		}
		
		if(position.y > bounds.getBottom()) {
			position.y = bounds.getBottom();
			returnVal = returnVal || true;
		}
		
		if(position.z > bounds.getNear()) {
			position.z =  bounds.getNear();
			returnVal = returnVal || true;
		}
		
		if(position.z < bounds.getFar()) {
			position.z =  bounds.getFar();
			returnVal = returnVal || true;
		}

		return returnVal;


	}

}
