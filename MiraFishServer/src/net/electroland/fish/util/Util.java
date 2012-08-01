package net.electroland.fish.util;

public class Util {
	public static float plusOrMinus(float val, float range ) {
		return val + range - ((float) (Math.random() * 2.0 * range));
	}

}
