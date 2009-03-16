package net.electroland.lighting.detector.animation;

public interface Animation 
{
	abstract public Raster getFrame();
	abstract public void initialize();
	abstract public void cleanUp();
	abstract public boolean isDone();
}