package net.electroland.lighting.detector.models;

import net.electroland.lighting.detector.DetectionModel;

/**
 * This class determines the average green intensity of an array of pixels, and
 * returns that value as a byte from 0-255.  It returns the same value for
 * getColor as getBrightness.
 * 
 * @author geilfuss
 *
 */
public class GreenDetectionModel implements DetectionModel {

	final public byte getValue(int[] pixels) {
		float g = 0;
		for (int i = 0; i < pixels.length; i++){
			g += (pixels[i] >> 8 & 0xFF);
		}
		return (byte)(g / pixels.length);
	}
}