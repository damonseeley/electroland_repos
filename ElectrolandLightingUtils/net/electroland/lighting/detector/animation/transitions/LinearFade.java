package net.electroland.lighting.detector.animation.transitions;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class LinearFade implements Animation {

	private long finishTime, startTime;
	private Raster raster;

	// should throw some kind of IllegalArgument exception if seconds <= 0;
	public LinearFade(double seconds, Raster raster)
	{
		this.finishTime = (long)(seconds * 1000);
		this.raster = raster;
	}
	
	public void cleanUp() {
		//  do nothing.
	}

	public Raster getFrame() {

		// calculate color
		double percentDone = (System.currentTimeMillis() - startTime) / 
							(double)(finishTime - startTime);
		int color = 255 - (int)(255 * percentDone);

		if (raster.isJava2d())
		{
			BufferedImage image = (BufferedImage)raster.getRaster();
			Graphics g = image.getGraphics();
			g.setColor(new Color(color,color,color));
			g.fillRect(0, 0, image.getWidth(), image.getHeight());
		}else
		{
			PGraphics image = (PGraphics)raster.getRaster();
			image.beginDraw();
			image.background(color);
			image.endDraw();
		}
		return raster;
	}

	public void initialize() {
		startTime = System.currentTimeMillis();
		finishTime += startTime;
	}

	public boolean isDone() {
		// stop when current time is later than finish time.
		return finishTime - System.currentTimeMillis() < 0;
	}
}
