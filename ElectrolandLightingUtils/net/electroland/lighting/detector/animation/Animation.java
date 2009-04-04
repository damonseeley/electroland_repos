package net.electroland.lighting.detector.animation;

public interface Animation
{
	abstract public boolean isDone();
	abstract public void initialize();
	abstract public Raster getFrame();
	abstract public void cleanUp();
}