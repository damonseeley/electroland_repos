package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;

public class NoEntryMap extends Constraint {
	boolean[][] map;

	public NoEntryMap(int priority, boolean[][] map) {
		super(priority);
		this.map = map;
	}

	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		try {
			if(map[(int)position.x][(int)position.y]) {
				if(! map[(int)b.getOldPosition().x][(int)b.getOldPosition().y]) {
					if(! map[(int)position.x][(int)  b.getOldPosition().y]) {
						position.y =  b.getOldPosition().y;
					} else if (! map[(int)b.getOldPosition().x][(int) position.y]) {
						position.x =  b.getOldPosition().x;
					} else {
						position.x = b.getOldPosition().x;
						position.y = b.getOldPosition().y;
					}
					
					
					return true;
				} else {
					// embeded in no entry region so sink to hide
					position.z = 0;
					
				}
			}
		} catch (RuntimeException e) { 
			// just incase runs off side of screen
			return false;
		}

		return false;
	}

}
