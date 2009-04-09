package net.electroland.noho.graphics.transitions;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;

import net.electroland.noho.graphics.ImageGenerator;
import net.electroland.noho.graphics.Transition;

public class GenericTransition extends Transition  {
	ImageGenerator mask;
	ImageGenerator overlay;
	public GenericTransition(int width, int height, ImageGenerator mask) {
		this(width, height, mask, null);
	}
	public GenericTransition(int width, int height, ImageGenerator mask, ImageGenerator overlay) {
		super(width, height);
		this.mask = mask;
		this.overlay = overlay;
	}

	@Override
	protected void renderTransition(long dt, long curTime, double interpVal, Graphics2D g2d) {
		mask.nextFrame(dt, curTime);
		if(channel1 != null) {
			channel1.ig.nextFrame(dt, curTime);
			Graphics2D chg2d = channel1.image.createGraphics();
			chg2d.setComposite(AlphaComposite.getInstance(AlphaComposite.DST_OUT, 1.0f));
			chg2d.drawImage(mask.getImage(), 0,0, null);			
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC, 1.0f));
			g2d.drawImage(channel1.image, 0, 0, null);
		}

		if(channel2 != null ) {
			channel2.ig.nextFrame(dt, curTime);
			if(channel1 != null) {
				Graphics2D chg2d = channel2.image.createGraphics();
				chg2d.setComposite(AlphaComposite.getInstance(AlphaComposite.DST_IN, 1.0f));
				chg2d.drawImage(mask.getImage(), 0,0, null);			
			}
			g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
			g2d.drawImage(channel2.image, 0, 0, null);
		}
		if(channel1 != null) {
			if(overlay != null) {
				overlay.nextFrame(dt, curTime);
				g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, 1.0f));
				g2d.drawImage(overlay.getImage(), 0, 0, null);			
			}
		}

	}
	
	public void reset() {
		mask.reset();
		if(overlay != null) overlay.reset();
	}

}
