package net.electroland.lighting.detector.models;

import net.electroland.lighting.detector.DetectionModel;

public class BrightRedDetectionModel implements DetectionModel {

	public byte getValue(int[] pixels) {
		int points = 0;
		float r = 0;
		for (int i = 0; i < pixels.length; i++){
			int intensity = (pixels[i] >> 16) & 0xFF;
			if (intensity > 0)
			{
				r += intensity;
				points++;
			}
		}
		return points == 0 ? (byte)0 : (byte)(r / points);
	}
}
