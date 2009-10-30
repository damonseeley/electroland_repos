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

	public void scale(double scalePositions, double scaleDimensions){
		this.x = (int)(this.x * scalePositions);
		this.y = (int)(this.y * scalePositions);
		this.width = (int)(this.width * scaleDimensions);
		this.height = (int)(this.height * scaleDimensions);
	}
	
	/**
	 * This is only here for LAFM, in a pinch.
	 * @deprecated
	 */
	public String getLightGroup(){
		return lightgroup;
	}
	
	public DetectionModel getModel() {
		return model;
	}
}