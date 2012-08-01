package net.electroland.fish.core;

import javax.vecmath.Vector3f;

public class ForceWeightPair {
	public Vector3f force;
	public float weight;
	
	public ForceWeightPair(Vector3f f, float w) {
		force = f;
		weight = w;
	}
	public ForceWeightPair() {
		
	}
	
}