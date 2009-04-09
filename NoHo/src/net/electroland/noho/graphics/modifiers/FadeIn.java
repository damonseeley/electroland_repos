package net.electroland.noho.graphics.modifiers;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.graphics.ImageModifyer;
import net.electroland.noho.util.DeltaTimeInterpolator;

/**
 * fades an image in
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class FadeIn extends ImageModifyer {
	boolean isDone = false;
	
	DeltaTimeInterpolator interp;
	
	Color color = new Color(0,0,0);
	

	public FadeIn(long fadeTime) {
		interp = new 	DeltaTimeInterpolator(fadeTime);
	}
	@Override
	public void modifyImage(long dt, long curTime, BufferedImage image) {
		if(! interp.isDone()) {
			Graphics2D g2d = image.createGraphics();
			int curAlpha = (int) (225 * ( interp.interp(dt)));
			g2d.setColor(new Color(0,0,0,curAlpha));
			g2d.fillRect(0, 0, image.getWidth(), image.getHeight());
		}  else { // done do nothing
			isDone = true;
		}
	}

	@Override
	public boolean isDone() {
		return isDone;
	}

}
