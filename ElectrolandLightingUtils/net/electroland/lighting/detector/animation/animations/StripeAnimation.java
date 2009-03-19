package net.electroland.lighting.detector.animation.animations;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.image.BufferedImage;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class StripeAnimation implements Animation {

	private Raster raster;
	private int x, width;
	private boolean isDone = false;

	public StripeAnimation(BufferedImage image, int width)
	{
		raster = new Raster(image);
		this.width = width;
	}

	public void stop(){
		isDone = true;
	}
	
	public void cleanUp()
	{
		// nothing to do.
	}

	public Raster getFrame()
	{
		BufferedImage i = (BufferedImage)raster.getRaster();
		Graphics g = i.getGraphics();
		g.setColor(Color.BLACK);
		g.fillRect(0, 0, i.getWidth(), i.getHeight());
		g.setColor(Color.WHITE);
		g.fillRect(i.getWidth()%(x++), 0, width, i.getHeight());
		return raster;
	}

	public void initialize()
	{
		x = 0;
	}

	public boolean isDone()
	{
		return isDone;
	}
}
