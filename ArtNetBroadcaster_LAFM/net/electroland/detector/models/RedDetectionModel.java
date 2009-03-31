package net.electroland.detector.models;

import net.electroland.detector.DetectionModel;

/**
 * This class determines the average red intensity of an array of pixels, and
 * returns that value as a byte from 0-255.  It returns the same value for
 * getColor as getBrightness.
 * 
 * @author geilfuss
 *
 */
public class RedDetectionModel implements DetectionModel {

	final public byte getValue(int[] pixels) {
//		System.out.print("red check -> ");
		float r = 0;
		for (int i = 0; i < pixels.length; i++){
//			System.out.print("pxl=" + pixels[i]);
//			System.out.print(", shft=");
//			System.out.print((pixels[i] >> 16 & 0xFF));
			r += (pixels[i] >> 16) & 0xFF;
//			System.out.print(", ");
		}
//		System.out.println("");
		return (byte)(r / pixels.length);
	}
}