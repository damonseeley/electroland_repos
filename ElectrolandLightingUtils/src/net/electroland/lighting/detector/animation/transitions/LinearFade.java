package net.electroland.lighting.detector.animation.transitions;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Properties;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;
import processing.core.PGraphics;

public class LinearFade implements Animation {

	private long finishTime, startTime;
	private Raster raster;

	public void init(Properties props){
		
	}
	
	public LinearFade(double seconds, Raster raster)
	{
		this.raster = raster;
		startTime = System.currentTimeMillis();
		finishTime = startTime + (long)(seconds * 1000);
	}

	public Raster getFrame() {

		// we've agreed on the convention that 0 = all first show, 255 = all second show.
		double percentDone = (System.currentTimeMillis() - startTime) / 
								(double)(finishTime - startTime);

		int color = (int)(255 * (percentDone > 1.0 ? 1.0 : percentDone));

		if (raster.isJava2d())
		{
			BufferedImage image = (BufferedImage)raster.getRaster();
			Graphics2D g = image.createGraphics();
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

	public boolean isDone() {
		// stop when current time is later than finish time.
		return finishTime - System.currentTimeMillis() < 0;
	}
}
