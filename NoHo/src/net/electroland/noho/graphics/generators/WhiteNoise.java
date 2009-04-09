package net.electroland.noho.graphics.generators;


import java.awt.Color;

import net.electroland.noho.graphics.ImageGenerator;

/**
 * generates white noise
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */
public class WhiteNoise extends ImageGenerator {

	public WhiteNoise(int width, int height) {
		super(width, height);
	}

	@Override
	public void render(long dt, long curTime) {
		for(int x=0; x < image.getWidth(); x++) {
			for(int y = 0; y < image.getHeight(); y++) {
				image.setRGB(x, y, new Color((float)Math.random(),(float)Math.random(),(float)Math.random(), (float)Math.random()).getRGB());
			}
		}
		
	}

	@Override
	public boolean isDone() {
		return false;
	}

	public void reset() {
		//nothing to do
	}



}
