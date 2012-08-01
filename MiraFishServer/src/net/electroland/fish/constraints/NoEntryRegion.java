package net.electroland.fish.constraints;

import javax.vecmath.Vector3f;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Constraint;
import net.electroland.fish.util.Bounds;


public class NoEntryRegion extends Constraint {
	Bounds bounds;
	public NoEntryRegion(int priority, Bounds bounds) {
		super(priority);
		this.bounds = bounds;
	}

	@Override
	public boolean modify(Vector3f position, Vector3f velocity, Boid b) {
		if(bounds.contains(position.x, position.y)) {
			if(bounds.contains(b.getOldPosition().x, b.getOldPosition().y)) { 
				float dTop = position.y- bounds.getTop();
				float dBot = bounds.getBottom()-position.y;
				float dLeft = position.x - bounds.getLeft();
				float dRight = bounds.getRight() - position.x;
				if(dLeft<dRight) {
					if(dTop<dBot) {
						if(dLeft<dTop) {
							position.x = bounds.getLeft() -1;
							return true;
						} else {
							position.y = bounds.getTop() -1;
							return true;
						}
					} else {
						if(dLeft<dBot) {
							position.x = bounds.getLeft() -1;
							return true;
						} else {
							position.y = bounds.getBottom() + 1;
							return true;
						}
					}
				} else {
					if(dTop<dBot) {
						if(dRight<dTop) {
							position.x = bounds.getRight() +1;
							return true;
						} else {
							position.y = bounds.getTop() -1;
							return true;
						}
					} else {
						if(dRight<dBot) {
							position.x = bounds.getRight() +1;
							return true;
						} else {
							position.y = bounds.getBottom() + 1;
							return true;
						}
					}					
				}
			} // already embeded nothing to do but let it swim out
			position.x = b.getOldPosition().x;
			position.y = b.getOldPosition().y;
			return true;
		}
		return true;
	}
}

/*

			float x = (position.x + b.getOldPosition().x) * .5f;
			float y = (position.y + b.getOldPosition().y) * .5f;



			while(bounds.contains(x,y)) {
				 x = (x + b.getOldPosition().x) * .5f;
				 y = (y + b.getOldPosition().y) * .5f;				
				 float diff = (x - b.getOldPosition().x);
				 diff *= diff;
				 if(diff <- 1) {
					 x = b.getOldPosition().x;
				 }
				 diff = (y - b.getOldPosition().y);
				 diff *= diff;
				 if(diff <- 1) {
					 y = b.getOldPosition().y;
				 }
			}

			position.x = x;
			position.y = y;
			return true;

		}

		return false;
		/*
		Vector3f p = position;
		if(bounds.contains(p.x, p.y)) {
			float minX = p.x - bounds.getLeft();
			float tmp = bounds.getRight()-p.x;
			boolean leftMin = true;
			if(tmp < minX) {
				leftMin = false;
				minX = tmp;
			}

			float minY = p.y - bounds.getTop();
			tmp = bounds.getBottom()-p.y;
			boolean topMin = true;
			if(tmp < minY) {
				topMin = false;
				minY = tmp;
			}

			if(minX < minY) {
				if(leftMin) {
					p.x = bounds.getLeft();
					return true;
				} else {
					p.x = bounds.getRight();					
					return true;
				}
			} else if (topMin) {
				p.y = bounds.getTop();
				return true;
			} else {
				p.y = bounds.getBottom();
				return true;
			}



		} else {
			return false;
		}*/
//}

//}
