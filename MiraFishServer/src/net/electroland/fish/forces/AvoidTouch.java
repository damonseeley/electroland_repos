package net.electroland.fish.forces;

import java.awt.Point;

import net.electroland.fish.core.Boid;
import net.electroland.fish.core.Force;
import net.electroland.fish.core.ForceWeightPair;
import net.electroland.fish.core.SpacialGrid.Cell;

public class AvoidTouch extends Force {
	float strength;
	float strengthRoot22;
	public AvoidTouch(float weight, Boid self, float strength) {
		super(weight, self);
		this.strength = strength;
		this.strengthRoot22 = strength * .707f;
		returnValue.weight = weight;
	}

	@Override
	public ForceWeightPair getForce() {
		if(! self.isTouched) {
			Point pt = new Point(self.getScpacialGridLoc());
			
			pt.x = (pt.x < 1) ? 1 : pt.x;
			pt.x = (pt.x > self.pond.grid.cells.length-2) ? self.pond.grid.cells.length-2 : pt.x;

			pt.y = (pt.y < 1) ? 1 : pt.y;
			pt.y = (pt.y > self.pond.grid.cells[0].length-2) ? self.pond.grid.cells[0].length-2 : pt.y;

			Cell[] col;
			
			col = self.pond.grid.cells[pt.x -1]; // left
			if(col[pt.y - 1].isTouched) { // top
				returnValue.force.set(strengthRoot22,strengthRoot22,0);
				return returnValue;
			} else  if(col[pt.y].isTouched) {
				returnValue.force.set(strength,0,0);
				return returnValue;
			} else  if(col[pt.y+1].isTouched) {
				returnValue.force.set(strengthRoot22,-strengthRoot22,0);
				return returnValue;
			}
			col = self.pond.grid.cells[pt.x +1];
			if(col[pt.y - 1].isTouched) {
				returnValue.force.set(-strengthRoot22,strengthRoot22,0);
				return returnValue;
			} else  if(col[pt.y].isTouched) {
				returnValue.force.set(-strength,0,0);
				return returnValue;
			} else  if(col[pt.y+1].isTouched) {
				returnValue.force.set(-strengthRoot22,-strengthRoot22,0);
				return returnValue;
			}
			col = self.pond.grid.cells[pt.x];
			if(col[pt.y - 1].isTouched) {
				returnValue.force.set(0, strength,0);
				return returnValue;
			} else  if(col[pt.y+1].isTouched) {
				returnValue.force.set(0, -strength,0);
				return returnValue;
			}
			
		}
		return ZERORETURNVALUE;
	}

}
