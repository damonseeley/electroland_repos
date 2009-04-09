package net.electroland.noho.graphics;

import java.awt.AlphaComposite;
import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

/**
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
abstract public class ImageGenerator {
	
	protected BufferedImage image;
	private ImageConsumer consumer;
	
	
	protected Color bgColor = null;

	
	
	/**
	 * @param width
	 * @param height
	 */
	public ImageGenerator(int width, int height) {
		image = new BufferedImage(width, height, BufferedImage.TYPE_INT_ARGB);
	}
	
	public void clearBackground(Graphics2D g2d) {
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.CLEAR, 1.0f));
		g2d.setColor(Color.BLACK);
		g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
	}
	
	public void drawBackground(Graphics2D g2d) {
		if(bgColor != null) {
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			g2d.setColor(bgColor);
			g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
		}
	}
	
	public void setBackgroundColor(Color c) {
		bgColor = c;
	}
	
	
	/**
	 * Overide to generate image.  ImageGenerators should draw it the BufferedImage image. 
	 * Do not assume that image is unchanged between frames.  This method should not be called directly (except by nextFrame)
	 * @param dt - elapsed time (in milliseconds) since last call to render
	 */
	abstract protected void render(long dt, long curTime);
	
	
	/**
	 * Call to render next frame.  Consumer must be set (by calling setConsumer) before calling nextFrame.
	 * @param dt - elapsed time (in milliseconds) since last call to nextFrame
	 */
	public void nextFrame(long dt, long curTime) {
		render(dt, curTime);
		if(consumer != null)
			consumer.renderImage(dt, curTime, image);
	}
	
	/**
	 * 
	 * @param consumer - consumer to which generated images should be sent
	 */
	public void setConsumer(ImageConsumer consumer) {
		this.consumer = consumer;
	}
	
	public ImageConsumer getConsumer() {
		return consumer;
		
	}
	
	public BufferedImage getImage() { return image; }
	
	abstract public boolean isDone();

	abstract public void reset();
}

