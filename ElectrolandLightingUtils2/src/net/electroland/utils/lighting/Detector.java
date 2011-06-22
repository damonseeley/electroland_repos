package net.electroland.utils.lighting;

import java.util.Vector;

public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;
	protected Vector<Integer>pixels;

	protected boolean isOn;
	
	public void setState(boolean isOn){
		this.isOn = isOn;
	}
	public int getX() {
		return x;
	}
	public int getY() {
		return y;
	}
	public int getWidth() {
		return width;
	}
	public int getHeight() {
		return height;
	}
	public DetectionModel getModel() {
		return model;
	}
	
	public CanvasDetector evaluate(){
		return null;
	}
}
