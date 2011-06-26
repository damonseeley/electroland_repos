package net.electroland.utils.lighting;

import java.awt.Shape;
import java.util.ArrayList;

public class CanvasDetector {

	protected Shape boundary;
	protected DetectionModel detectorModel;
	protected byte latestState;
	protected ArrayList<String> tags;
	protected ArrayList<Integer>pixelIndices = new ArrayList<Integer>(); // TODO: this seems like a funny place for this.
	
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
	public ArrayList<Integer> getPixelIndices() {
		return pixelIndices;
	}
	
	public String toString()
	{
		StringBuffer sb = new StringBuffer("CanvasDetector[");
		sb.append(boundary).append(',');
		sb.append("tags[").append(tags).append("],latestEval=");
		sb.append(latestState).append(',').append(detectorModel).append(']');
		return sb.toString();
	}
}