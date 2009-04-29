package net.electroland.laface.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class ImageSequence implements Animation {

	private Raster r;
	private int index = 0;
	private PImage[] sequence;
	
	public ImageSequence(Raster r, PImage[] sequence){
		this.r = r;
		this.sequence = sequence;
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
			c.image(sequence[index++], 0, 0);
			c.endDraw();
		}
		return r;
	}
	
	public void cleanUp() {
	}

	public boolean isDone() {
		if (index == sequence.length){
			return true;
		}
		return false;
	}

}
