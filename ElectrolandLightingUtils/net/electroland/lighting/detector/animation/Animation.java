package net.electroland.lighting.detector.animation;

public interface Animation
{
	abstract public boolean isDone();
	abstract public void initialize(); // this should go away.
	abstract public Raster getFrame();
	abstract public void cleanUp(); // this should return a Raster.
	//abstract public String getID();
}