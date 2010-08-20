package net.electroland.lighting.detector.animation.animations;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Properties;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Throb implements Animation {

	private int i, d, c;
	private boolean isDone = false;
	private Raster raster;
	private Properties props;

	public Throb(BufferedImage image)
	{
		raster = new Raster(image);
		c = 2;
		i = 0;
		d = 20;
	}

	public void init(Properties props){
		this.props = props;
	}

	public boolean isDone() 
	{
		return isDone || c < 0;
	}

	public void stop()
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
			c--;
			i = 0;
			d = 20;
		}

		Graphics2D g = b.createGraphics();
		g.setColor(new Color(i,i,i));
		g.fillRect(0, 0 , b.getWidth(), b.getHeight());
		return raster;
	}
}