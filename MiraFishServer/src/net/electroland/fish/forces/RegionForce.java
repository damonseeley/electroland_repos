package net.electroland.fish.forces;

import java.awt.Rectangle;
import java.util.Vector;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.util.RegionMap;

public class RegionForce extends Force {
	protected Vector<MaskForcePair> forces = new Vector<MaskForcePair>();
	protected RegionMap map;
	int masterKey;
	Vector3f vec3f = new Vector3f(0,0,0);


	public RegionForce(float weight, Boid self, RegionMap regionMap) {
		super(weight, self);
		map = regionMap;
		returnValue.force.set(0,0,0);
		returnValue.weight = 0;
	}

	@Override
	public ForceWeightPair getForce() {
		Vector3f pos = self.getPosition();
		return(getForce((int)pos.x, (int)pos.y));
	}

	
	public ForceWeightPair getForce(int x, int y) { // use for shared maps 
		int bits = map.getRegions(x,y);
		if((bits | masterKey) != 0) { // if in at least one region
			returnValue.force.set(0,0,0);
			returnValue.weight = 0;
			for(MaskForcePair pair : forces ) {
				vec3f.add(pair.getForce(bits));
				returnValue.weight = weight;
			}
		}
		return returnValue;
	}
	public static final Vector3f ZERO = new Vector3f(0,0,0);
	
	public boolean addForce(Rectangle r, Vector3f f) {
		int mask = map.addRegion(r);
		if(mask == 0) return false;
		forces.add(new MaskForcePair(mask, f));
		return true;
	}

	public class MaskForcePair {
		public int mask;
		public Vector3f force;
		public MaskForcePair(int mask, Vector3f force) {
			this.mask = mask;
			this.force = force;
			masterKey |= mask;
		}
		
		public Vector3f getForce(int bits) { 
			if((bits | mask) != 0) {
				return force;
			} else {
				return ZERO;
			}
		}
	}

}
