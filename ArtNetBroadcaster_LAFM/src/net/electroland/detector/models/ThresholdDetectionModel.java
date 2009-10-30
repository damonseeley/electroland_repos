package net.electroland.detector.models;

import net.electroland.detector.DetectionModel;

public class ThresholdDetectionModel implements DetectionModel {

	private float threshold = .5f; // default is a 50% threshold
	
	public float getThreshold() {
		return threshold;
	}

	public void setThreshold(float threshold){
		if (threshold < 0 || threshold > 1.0){
			throw new RuntimeException("Threshold must be between 0 and 1.0");
		}else{
			this.threshold = threshold;			
		}
	}

	public byte getValue(int[] pixels) {
		float total = 0;
		for (int i = 0; i < pixels.length; i++){
			if (pixels[i] != 0){ // might have to actually do some bit checking here, to ignore alpha.
				total += 1;
			}
		}
		return (total / pixels.length) >= threshold ? (byte)255 : (byte)0;
	}
}