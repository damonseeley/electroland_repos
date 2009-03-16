package net.electroland.lighting.detector.animation.animations;

import java.awt.Color;
import java.awt.image.BufferedImage;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Throb implements Animation {

	private int i, d;
	private boolean isDone = false;
	private Raster raster;

	public Throb(BufferedImage image)
	{
		raster = new Raster(image);
		i = 0;
		d = 20;
	}

	public boolean isDone() 
	{
		return isDone;
	}

	public void done()
	{
		isDone = true;
	}
	
	public Raster getFrame() 
	{
		BufferedImage b = (BufferedImage)raster.getRaster();

		i += d;
		if (i > 255)
		{
			i = 255;
			d = -20;
		}else if (i < 0)
		{
			i = 0;
			d = 20;
		}

		b.getGraphics().setColor(new Color(i,i,i));
		b.getGraphics().fillRect(0, 0 , b.getWidth(), b.getHeight());
		
		return raster;
	}

	public void cleanUp() {
		// do nothing.
	}
	public void initialize() {
		// do nothing.
	}	
}