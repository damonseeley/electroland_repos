package net.electroland.noho.graphics;

import java.awt.image.BufferedImage;

/**
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public interface ImageConsumer {
	
	/**
	 * @param dt - time in milliseconds since last call to renderImage
	 * @param image - image to use (and possibly modify)
	 */
	public void renderImage(long dt, long curTime, BufferedImage image);
}
