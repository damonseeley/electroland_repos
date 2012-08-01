package net.electroland.fish.constraints;

import java.awt.Polygon;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;

public class NoEntryPoly extends Constraint {
	Polygon poly;

	public NoEntryPoly(int priority, Polygon poly) {
		super(priority);
		this.poly = poly;
	}

	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		if(poly.contains(position.x, position.y)) {
			if(poly.contains(b.getOldPosition().x, b.getOldPosition().y)) { return false; } // already embeded nothing to do but let it swim out

			float x = (position.x + b.getOldPosition().x) * .5f;
			float y = (position.y + b.getOldPosition().y) * .5f;
			
			if(poly.contains(x,y)) { // if the half point is in bounds just go to the old point
				position.x = b.getOldPosition().x;
				position.y = b.getOldPosition().y;
				return true;
			}
			position.x = x;
			position.y = y;
			return true;
			
		}
		return false;
	}

}
