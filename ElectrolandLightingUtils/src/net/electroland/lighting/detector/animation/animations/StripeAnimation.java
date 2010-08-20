package net.electroland.lighting.detector.animation.animations;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Properties;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class StripeAnimation implements Animation {

	private Raster raster;
	private int x, width;
	private boolean isDone = false;

	public StripeAnimation(BufferedImage image, int width)
	{
		x = 0;
		raster = new Raster(image);
		this.width = width;
	}

	public void init(Properties props){
		
	}

	public void stop(){
		isDone = true;
	}

	public Raster getFrame()
	{
		BufferedImage i = (BufferedImage)raster.getRaster();
		Graphics2D g = i.createGraphics();
		g.setColor(Color.BLACK);
		g.fillRect(0, 0, i.getWidth(), i.getHeight());
		g.setColor(Color.WHITE);
		g.fillRect(i.getWidth()%(x++), 0, width, i.getHeight());
		return raster;
	}

	public boolean isDone()
	{
		return isDone;
	}
}