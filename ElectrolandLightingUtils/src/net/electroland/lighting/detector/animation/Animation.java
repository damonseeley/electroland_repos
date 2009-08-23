package net.electroland.lighting.detector.animation;

public interface Animation
{
	abstract public boolean isDone();
	abstract public Raster getFrame();
}