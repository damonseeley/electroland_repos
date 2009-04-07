package net.electroland.laface.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Highlighter implements Animation {

	private Raster r;
	
	public Highlighter(Raster r){
		this.r = r;
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			for(int i=0; i<c.width; i+=50){
				c.fill(255);
				c.rect(i, 0, 25, c.height);	// TODO test the transition effect
			}
			c.endDraw();
		}
		return r;
	}
	
	public void cleanUp() {
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return false;
	}

}
