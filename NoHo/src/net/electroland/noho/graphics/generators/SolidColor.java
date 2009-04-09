package net.electroland.noho.graphics.generators;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.image.BufferedImage;

import net.electroland.noho.graphics.ImageGenerator;

/**
 * generates a solid color
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class SolidColor extends ImageGenerator {
	int timeLeft;
	int holdTime;
	BufferedImage referenceImage;

	
	public SolidColor(int width, int height, Color c, int holdTime) {
		super(width, height);
		this.holdTime = holdTime;
		timeLeft = holdTime;
		setBackgroundColor(c);
	}

	@Override
	public void render(long dt, long curTime) {
		timeLeft -= dt;
		Graphics2D g2d = image.createGraphics();
		clearBackground(g2d);
		drawBackground(g2d);
	}

	@Override
	public boolean isDone() {
		return timeLeft <= 0;
	}

	public void reset() {
		timeLeft = holdTime;
	}



}
