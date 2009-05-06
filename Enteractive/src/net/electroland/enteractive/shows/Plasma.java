package net.electroland.enteractive.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Plasma implements Animation {

	private Raster r;
	private float xc, yc, calc1, calc2, s, s1, s2, s3;
	private int gridx, gridy, frameCount, element;
	private int xsize, ysize;
	
	public Plasma(Raster r){
		this.r = r;
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.HSB, 255, 255, 255, 255);
		gridx = raster.width;
		gridy = raster.height;
		frameCount = 0;
		xsize = 5;
		ysize = 5;
	}

	public Raster getFrame() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.HSB, 255, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);		// clear the raster
		
		xc = 25;
		frameCount += 1;
		calc1 = (float)Math.sin(Math.toRadians(frameCount* 0.61655617));
		calc2 = (float)Math.sin(Math.toRadians(frameCount* -3.6352262));
		
		raster.loadPixels();
		for(int x=0; x<gridx; x++, xc+=xsize){
			yc = 25;
			s1 = (float) (128 + 128 * Math.sin(Math.toRadians(xc) * calc1));
			for(int y=0; y<gridy; y++, yc+=ysize){
				s2 = (float)(128 + 128 * Math.sin(Math.toRadians(yc) * calc2));
				s3 = (float)(128 + 128 * Math.sin(Math.toRadians((xc + yc + frameCount * 10)/2)));
				s = (s1 + s2 + s3) / 3;
				element = x+y*gridx;
				raster.pixels[element] = raster.color((int)s, 255 - (int)s / 2.0f, 255);
				//raster.pixels[element] = (int)s2;
			}
		}
		raster.updatePixels();
		raster.endDraw();
		return r;
	}

	public boolean isDone() {
		return false;
	}
	
}
