package net.electroland.fish.core;

import javax.vecmath.Vector3f;

import net.electroland.fish.util.ImageLoader;

public class Maps {
	public  static boolean[][] NO_ENTRY;

	public  static Vector3f[][] FORCES;

	
	public static void loadMaps(String entryMap, String rotateMap, float rotateWeight, String edgeMap, float edgeWeight) {
		NO_ENTRY = ImageLoader.getBoolArrayForImageFile(entryMap);
		FORCES = ImageLoader.getVecArrayForImageFile(rotateMap);
		ImageLoader.mergVecArrayForImageFile(edgeMap, FORCES, rotateWeight, edgeWeight);
	}

}
