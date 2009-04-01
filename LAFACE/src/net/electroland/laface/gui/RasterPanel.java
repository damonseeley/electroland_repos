package net.electroland.laface.gui;

import net.electroland.lighting.detector.animation.Raster;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PImage;

/**
 * Displays the raster image as well as allows toggling of lights directly.
 * @author asiegel
 */

@SuppressWarnings("serial")
public class RasterPanel extends PApplet {
	
	private Raster raster;
	private int guiWidth, guiHeight;
	
	public RasterPanel(int width, int height, Raster raster){
		this.raster = raster;
		if(raster.isProcessing()){
			PImage c = (PImage)raster.getRaster();
			guiWidth = c.width;
			guiHeight = c.height;
		} else {
			guiWidth = 320;
			guiHeight = 240;
		}
	}
	
	public void setup(){
		size(guiWidth, guiHeight);
		frameRate(30);
		noStroke();
		colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	
	public void draw(){
		background(0);
		noFill();
		stroke(50);
		if(raster.isProcessing()){
			PImage image = (PImage)raster.getRaster();
			image(image, 0, 0);
		}
	}
	
	public void mouseDragged(){
		// check for light object over raster to turn on/off
	}
	
	public void mousePressed(){
		// check for light object over raster to turn on/off
	}
	
	
	
	
	
	public class Light{
		
		private int x, y, width, height, brightness;
		
		public Light(int x, int y, int width, int height){
			this.x = x;
			this.y = y;
			this.width = width;
			this.height = height;
			brightness = 0;
		}
	}

}
