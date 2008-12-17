package net.electroland.lafm.animation;

public interface Animation {
	
	/**
	 * TODO: Animation should use PApplet's drawing methods to create a raster output to be fed to Flower.
	 */
	
	public void start();
	public void stop();
	public int[] draw();
	public int getDuration();
}
