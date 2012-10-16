package net.electroland.gotham.processing.assets;

public interface MoveBehavior {
	public static final int offset = 600;
	
	public void move();
	public float getPosition();
	public void setPosition(float x);
	public float getDist();
	public float getTimeAcross();
	public float getTarget();
	public float getBegin();
	public void pause();
	public void resume();
	public boolean pauseState();
}
