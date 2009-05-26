package net.electroland.laface.shows;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
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
		//System.out.println("video playing");
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			try {
				//System.out.println("getting mosaics...");
				BufferedImage[] imgs = lafvp.getMosaics();
				//System.out.println("mosaics received");
				if(imgs != null) {
					/*
					for(BufferedImage bi : imgs) {
						//System.out.println("image displayed");
						c.image(new PImage(bi),0,0,c.width,c.height);
					}
					*/
					// this is set up to overlap multiple mosaics and blend their light areas
					for(int i=0; i<imgs.length; i++){
						if(i == 0){
							c.image(new PImage(imgs[i]),0,0,c.width,c.height);
						} else {
							c.blend(new PImage(imgs[i]),0,0,c.width,c.height,0,0,c.width,c.height,PConstants.LIGHTEST);
						}
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
			c.endDraw();
		} else {
			// JAVA2D section for returning a buffered image
			Image c = (Image)(r.getRaster());
			try {
				//System.out.println("getting mosaics...");
				BufferedImage[] imgs = lafvp.getMosaics();
				//System.out.println("mosaics received");
				if(imgs != null) {
					for(int i=0; i<imgs.length; i++){
						// TODO this does not blend mosaics yet
						System.out.println(c.getWidth(null) +" "+ c.getHeight(null));
						Image scaledimg = imgs[i].getScaledInstance(c.getWidth(null), c.getHeight(null), Image.SCALE_FAST);
						c.getGraphics().drawImage(scaledimg, 0, 0, null);
					}
				}
			} catch (InterruptedException e) {
				e.printStackTrace();
			}
		}
		return r;
	}

	public boolean isDone() {
		return false;
	}

}
