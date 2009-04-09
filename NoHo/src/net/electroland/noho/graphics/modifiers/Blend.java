package net.electroland.noho.graphics.modifiers;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;
import java.util.Enumeration;
import java.util.Vector;

import net.electroland.noho.graphics.ImageConsumer;
import net.electroland.noho.graphics.ImageGenerator;

/**
 * blends two image
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class Blend extends ImageGenerator implements ImageConsumer {
	Vector<BufferedImage> images = new Vector<BufferedImage>(); 
	
	public Blend(int width, int height) {
		super(width, height);
	}

	@Override
	public void render(long dt, long curTime) {
		Graphics2D g2d = image.createGraphics();
		

		if(! images.isEmpty()) {
			Enumeration<BufferedImage> e = images.elements();
			g2d.drawImage(e.nextElement(), 0, 0, null);
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f / images.size()));
			while(e.hasMoreElements()) {
				g2d.drawImage(e.nextElement(), 0, 0, null);				
			}
		}
		images.clear();
	}

	public void renderImage(long dt, long curTime, BufferedImage image) {
		images.add(image);
	}

	@Override
	public boolean isDone() {
		return false;
	}

	public void reset() {
		// nothing to do
	}


	
	
}
