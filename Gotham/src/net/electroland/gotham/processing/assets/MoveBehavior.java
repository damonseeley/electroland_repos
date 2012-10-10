package net.electroland.gotham.processing.assets;

public interface MoveBehavior {
	public void move();
	public float getPosition();
	public void setPosition(float x);
	public float getTarget();
	public float getBegin();
}
