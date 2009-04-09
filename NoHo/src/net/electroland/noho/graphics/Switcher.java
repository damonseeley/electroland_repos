package net.electroland.noho.graphics;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.util.DeltaTimeInterpolator;

/**
 * Switcher performs crossfades between two ImageGenerators.  Is used by both BasicText and Compositor.
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */

public class Switcher extends ImageGenerator {
	
	

	Channel channel1 = null;
	Channel channel2 = null;
	
	DeltaTimeInterpolator interp = new DeltaTimeInterpolator(0);
	
	

	public class Channel implements ImageConsumer {
		BufferedImage image;
		ImageGenerator ig;
		public Channel(ImageGenerator ig) {
			this.ig = ig;
			ig.setConsumer(this);
		}

		public void renderImage(long dt, long curTime, BufferedImage image) {
			this.image = image;
		}
		
	}

	/**
	 * Start crossfade
	 * @param time - amount of time to interpolate
	 */
	public void startSwitch(long time) {
		interp.reset(time);
	}
	
	/**
	 * 
	 * @param ig - new imagegenerator to which to crossfade
	 */

	public void addImage(ImageGenerator ig) {
		if(ig != null) {
			channel2 = new Channel(ig);
		} else {
			channel2 = null;
		}
	}

	public void addImage(ImageGenerator ig, int channel) {
			if(channel == 1) {
				channel1 = new Channel(ig);

			} else {
				channel2 = new Channel(ig);
			}
	}

	
	
	public Switcher(int width, int height) {
		super(width, height);
	}

	/**
	 *  @return is done crossfading
	 */
	public boolean isDone() {
		return interp.isDone();
	}

	private void drawChannel(long dt, long curTime, Graphics2D g2d, Channel c) {
		if(c != null) {
			c.ig.nextFrame(dt, curTime);
			g2d.drawImage(c.image, 0, 0, null);
		} 
	}
	

	
	
	@Override
	protected void render(long dt, long curTime) {
		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		drawBackground(g2d);
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));

		if(interp.isDone()) {
			drawChannel(dt, curTime, g2d, channel1);
		} else {
			float alpha = (float) interp.interp(dt);
			if(interp.isDone()) { // if just finished
				channel1 = channel2;
				channel2 = null;
				drawChannel(dt, curTime, g2d, channel1);
			} else {
				g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float) alpha));
				drawChannel(dt, curTime, g2d, channel1);

				g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (float)(1.0 - alpha)));
				drawChannel(dt, curTime, g2d, channel2);
				
			}
		}


	}
	
	public void reset() {
		System.err.println("Cannot reset switcher");
	}



}
