package net.electroland.utils.lighting;

import java.awt.Shape;
import java.util.ArrayList;

public class CanvasDetector {

	protected Shape boundary;
	protected DetectionModel detectorModel;
	protected byte latestState;
	protected ArrayList<String> tags;
	protected ArrayList<Integer>indices = new ArrayList<Integer>(); // TODO: this seems like a funny place for this.
	
	public Shape getBoundary() {
		return boundary;
	}
	public DetectionModel getDetectorModel() {
		return detectorModel;
	}
	public byte getLatestState() {
		return latestState;
	}
	public void setEvaluatedValue(byte b){
		latestState = b;
	}
	public ArrayList<String> getTags() {
		return tags;
	}
	public ArrayList<Integer> getIndices() {
		return indices;
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer("CanvasDetector[");
		sb.append(tags).append(',').append(tags);
		sb.append(boundary).append(']');
		return sb.toString();
	}
}