package net.electroland.laface.shows;

import java.awt.image.BufferedImage;
import java.io.IOException;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.elvisVideoProcessor.ElProps;
import net.electroland.elvisVideoProcessor.LAFaceVideoProcessor;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Video implements Animation{
	
	private Raster r;
	private LAFaceVideoProcessor lafvp;
	
	public Video(Raster r, LAFaceVideoProcessor lafvp){
		this.r = r;
		this.lafvp = lafvp;
	}

	public Raster getFrame() {
		System.out.println("video playing");
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			try {
				System.out.println("getting mosaics...");
				BufferedImage[] imgs = lafvp.getMosaics();
				System.out.println("mosaics received");
				if(imgs != null) {
					for(BufferedImage bi : imgs) {
						System.out.println("image displayed");
						c.image(new PImage(bi),0,0,c.width,c.height);
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			c.endDraw();
		}
		return r;
	}

	public boolean isDone() {
		return false;
	}

}
