package net.electroland.lighting.detector.animation;

import java.awt.Image;

import processing.core.PImage;

public class Raster
{
	private Object raster;

	public Raster(Image image)
	{
		this.raster = image;
	}

	public Raster(PImage pimage)
	{
		this.raster = pimage;
	}

	final public boolean isJava2d()
	{
		// should cover Image AND BufferedImage
		return raster instanceof Image;
	}

	final public boolean isProcessing()
	{
		// should cover PImage AND PGraphics
		return raster instanceof PImage;
	}

	final public Object getRaster()
	{
		return raster;
	}
}