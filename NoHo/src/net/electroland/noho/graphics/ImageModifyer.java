package net.electroland.noho.graphics;

import java.awt.image.BufferedImage;

/**
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
abstract public  class ImageModifyer implements ImageConsumer {
	protected ImageConsumer consumer;
	
	/**	  
	 * @param consumer - consumer to which generated images should be sent
	 */
	public void setConsumer(ImageConsumer consumer) {
		this.consumer = consumer;
	}

	/**
	 * called by other ImageGenerators or ImageModyers
	 * @param dt - time in milliseconds since last call to renderImage
	 * @param image - image to use (and possibly modify)
	 */
	public void renderImage(long dt, long curTime, BufferedImage image) {
		modifyImage(dt, curTime, image);
		consumer.renderImage(dt, curTime, image);	
	}
	
	/**
	 * Overide to modify image
	 * @param dt - elapsed time since last call to modifyImage
	 * @param curTime TODO
	 * @param image - image ot modify
	 */
	abstract public void modifyImage(long dt, long curTime, BufferedImage image) ;
	abstract public boolean isDone();

}
