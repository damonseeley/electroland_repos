package net.electroland.fish.core;

import javax.vecmath.Vector2f;

// perserves 2d dirction safe
public class Vector2fDirPreserver extends Vector2f {

	private static final long serialVersionUID = 1L;
	
	public Vector2fDirPreserver(float x, float y) {
		super(x,y);
	}
	
	public boolean isNegative() {
		return (x * y) < 0;		
	}
	
	public void normalizeDirSafe() {
		boolean xneg = x < 0;
		boolean yneg = y < 0;
		super.normalize();
		if(xneg) {
			x = -x;
		}
		if(yneg) {
			y = -y;
		}
	}
	
	public void perp() {
		float tmp = x;
		x = y;
		y= -tmp;
		
	}

}
