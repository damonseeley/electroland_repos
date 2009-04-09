package net.electroland.noho.graphics;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.graphics.transitions.Wipe;


/**
 * Compositor has three layers - a background, forground, and overlay.
 * Adding an imageGenerator to a layer results in a crossfade (use a transition time of 0 for immediate transitions)
 * layers can be turned off if performance becomes a problem
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class Compositor extends ImageGenerator implements ImageConsumer {
	protected Transition background;
	protected Transition foreground;
	protected Transition overlay;

	protected boolean isForegroundEnabled = true;
	protected boolean isBackgroundEnabled = true;
	protected boolean isOverlayEnabled = true;
	
	public Compositor(int width, int height) {
		super(width, height);
		background = new Wipe(width, height);
		foreground = new Wipe(width, height);
		overlay = new Wipe(width, height);
		
		background.setConsumer(this);
		foreground.setConsumer(this);
		overlay.setConsumer(this);
	}
	
	public void setForgroundTransition(Transition newTrans) {
		swapTransition(newTrans, foreground);
		foreground = newTrans;
	}
	public void setBackgrondTransition(Transition newTrans) {
		swapTransition(newTrans, background);
		background = newTrans;
	}
	public void setOverlayTransition(Transition newTrans) {
		swapTransition(newTrans, overlay);
		overlay = newTrans;
	}

	public void swapTransition(Transition newTrans, Transition oldTrans) {
		newTrans.channel1 = oldTrans.channel1;
		newTrans.channel2 = oldTrans.channel2;
		newTrans.interp = oldTrans.interp;
		newTrans.setConsumer(this);
	}
	public void addForground(ImageGenerator ig, long transTime) {
		 addLayer(foreground, ig, transTime);
	}

	public void addBackground(ImageGenerator ig, long transTime) {
		 addLayer(background, ig, transTime);
	}

	public void addOverlay(ImageGenerator ig, long transTime) {
		 addLayer(overlay, ig, transTime);
	}

	public void isForegroundEnabled(boolean b) {
		isForegroundEnabled =b;
	}
	public void isBackgroundEnabled(boolean b) {
		isBackgroundEnabled =b;
	}
	public void isOverlayEnabled(boolean b) {
		isOverlayEnabled =b;
	}
	private void addLayer(Transition switcher, ImageGenerator ig, long transTime) {
		switcher.addImage(ig);
		switcher.startSwitch(transTime);
	}
	
	@Override
	public boolean isDone() {
		return false;
	}
	
	

	@Override
	protected void render(long dt, long curTime) {
		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		drawBackground(g2d);
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
		
		if(isBackgroundEnabled) {
			background.nextFrame(dt, curTime);
		}
		if(isForegroundEnabled) {
			foreground.nextFrame(dt, curTime);
		}
		if(isOverlayEnabled) {
			overlay.nextFrame(dt, curTime);
		}
		
	}

	public void renderImage(long dt, long curTime, BufferedImage image) {
		this.image.createGraphics().drawImage(image, 0, 0, null);
		
	}
	
	public void reset() {
		System.err.println("Cannot reset switcher");
	}


}
