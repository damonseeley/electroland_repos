package net.electroland.lighting.detector;

/**
 * represents a zone on a PImage that will be detected for transmission to a light fixture.
 * 
 * @author geilfuss
 */

public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;
	
	// hacky crap that should be removed:
	protected String patchgroup;
	protected int channel;

	public Detector(int x, int y, int width, int height, DetectionModel model){
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.model = model;
	}

	public Detector(int x, int y, int width, int height, DetectionModel model, String patchgroup, int channel){
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.model = model;
		this.patchgroup = patchgroup;
		this.channel = channel;
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
	
	public String getPatchgroup(){
		return patchgroup;
	}
	
	public DetectionModel getModel() {
		return model;
	}
}