package net.electroland.detector;

/**
 * represents a zone on a PImage that will be detected for transmission to a light fixture.
 * 
 * @author geilfuss
 */

public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;
	
	// hacky crap that should be removed:
	protected String lightgroup;
	protected int channel;

	public Detector(int x, int y, int width, int height, DetectionModel model){
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.model = model;
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
}