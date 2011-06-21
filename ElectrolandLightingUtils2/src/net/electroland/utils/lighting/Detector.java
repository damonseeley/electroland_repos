package net.electroland.utils.lighting;

import net.electroland.utils.lighting.DetectionModel;

public class Detector {

	protected int x, y, width, height;
	protected DetectionModel model;

	public int getX() {
		return x;
	}
	public void setX(int x) {
		this.x = x;
	}
	public int getY() {
		return y;
	}
	public void setY(int y) {
		this.y = y;
	}
	public int getWidth() {
		return width;
	}
	public void setWidth(int width) {
		this.width = width;
	}
	public int getHeight() {
		return height;
	}
	public void setHeight(int height) {
		this.height = height;
	}
	public DetectionModel getModel() {
		return model;
	}
	public void setModel(DetectionModel model) {
		this.model = model;
	}
}
