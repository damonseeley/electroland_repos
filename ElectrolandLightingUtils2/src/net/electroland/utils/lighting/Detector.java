package net.electroland.utils.lighting;


/**
 * an ABSTRACT detector. it is in the coordinate space of a Fixture.
 * @author bradley
 *
 */
public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;

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
