package net.electroland.lighting.detector.animation;

public interface Animation extends Completable
{
	abstract public Raster getFrame();
}