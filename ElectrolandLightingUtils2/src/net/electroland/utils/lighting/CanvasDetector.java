package net.electroland.utils.lighting;

import java.awt.Shape;
import java.util.Vector;
import java.util.concurrent.CopyOnWriteArrayList;

public class CanvasDetector {

	protected Shape boundary;
	protected DetectionModel detectorModel;
	protected byte latestState;
	protected Vector tags;
	protected CopyOnWriteArrayList<Integer>indices;
	
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
	public Vector getTags() {
		return tags;
	}
	public CopyOnWriteArrayList<Integer> getIndices() {
		return indices;
	}
}