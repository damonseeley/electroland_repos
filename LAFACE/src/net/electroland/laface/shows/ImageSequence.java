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
	private boolean mirror;
	
	public ImageSequence(Raster r, PImage[] sequence, boolean mirror){
		this.r = r;
		this.sequence = sequence;
		this.mirror = mirror;
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public void initialize() {		
	}	

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			if(mirror){
				int imageWidth = sequence[index].width;
				int imageHeight = sequence[index].height;
				sequence[index].loadPixels();
				int[] pixeldata = new int[sequence[index].pixels.length];
				System.arraycopy(sequence[index].pixels, 0, pixeldata, 0, sequence[index].pixels.length);
				for(int w=0; w<imageWidth; w++){
					for(int h=0; h<imageHeight; h++){
						sequence[index].pixels[h*imageWidth + w] = pixeldata[(imageWidth - w - 1) + h * imageWidth];
					}
				}
				sequence[index].updatePixels();
				c.image(sequence[index], 0, 0, c.width, c.height);
				index++;
			} else {
				c.image(sequence[index++], 0, 0, c.width, c.height);
			}
			c.endDraw();
		}
		if (index == sequence.length){
			index = 0;
		}
		return r;
	}
	
	public void cleanUp() {
	}

	public boolean isDone() {
		//if (index == sequence.length){
			//return true;
		//}
		return false;
	}

}
