package net.electroland.detector;


/**
 * A detection model is an implementation that knows how to distill an array
 * of pixel data into a single value. See examples in net.electroland.detector.models
 * 
 * @author geilfuss
 */
public interface DetectionModel {
	public byte getValue(int[] b); 
}