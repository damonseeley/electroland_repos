package net.electroland.noho.graphics.transitions;

import java.awt.AlphaComposite;
import java.awt.Graphics2D;

import net.electroland.noho.graphics.Transition;

/**
 * Switcher performs crossfades between two ImageGenerators.  Is used by both BasicText and Compositor.
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */

public class Crossfade extends Transition {
	
	


	public Crossfade(int width, int height) {
		super(width, height);
	}

	@Override
	public void renderTransition(long dt, long curTime, double interpVal, Graphics2D g2d) {
		float alpha = (float) interpVal;
		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, alpha));
		drawChannel(dt, curTime, g2d, channel1);

		g2d.setComposite(AlphaComposite.getInstance(AlphaComposite.SRC_OVER, (1.0f - alpha)));
		drawChannel(dt, curTime, g2d, channel2);		
	}
	@Override
	public void reset() {
		interp.reset();
		
	}

}
