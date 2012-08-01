package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;
import net.electroland.fish.util.Bounds;
/*
 * This class is a hack for testing things don't use
 */
public class WorldWrap extends Constraint {
	Bounds bounds;

	public WorldWrap(Bounds bounds) {
		super(Integer.MAX_VALUE); // make sure applied last
		this.bounds = bounds;
	}
	
	
	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		boolean returnVal = false;

		if(position.x < bounds.getLeft()) {
			position.x = bounds.getRight()-1;
			returnVal = returnVal || true;
		}
		
		if(position.x > bounds.getRight()) {
			position.x = bounds.getLeft()+1;
			returnVal = returnVal || true;
		}

		if(position.y < bounds.getTop()) {
			position.y = bounds.getBottom()-1;
			returnVal = returnVal || true;
		}
		
		if(position.y > bounds.getBottom()) {
			position.y = bounds.getTop()+1;
			returnVal = returnVal || true;
		}
		
		if(position.z < bounds.getNear()) {
			position.z =  bounds.getFar()-.01f;
			returnVal = returnVal || true;
		}
		
		if(position.z > bounds.getFar()) {
			position.z =  bounds.getNear() + .01f;
			returnVal = returnVal || true;
		}

		return returnVal;


	}

}
