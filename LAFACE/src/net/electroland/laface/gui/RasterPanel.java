package net.electroland.laface.gui;

import java.util.Collection;
import java.util.Iterator;
import java.util.ListIterator;

import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
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
	private Collection<Recipient> recipients;
	private int guiWidth, guiHeight;
	private int faceWidth, faceHeight;		// light grid
	
	public RasterPanel(Collection<Recipient> recipients){
		this.recipients = recipients;
		guiWidth = 1048;
		guiHeight = 133;
	}
	
	public void setRaster(Raster raster){
		this.raster = raster;
	}
	
	public void setup(){
		size(guiWidth, guiHeight);
		frameRate(30);
		noStroke();
		colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	
	public void draw(){
		background(0,0,0);
		noFill();
		noStroke();
		if(raster != null && raster.isProcessing()){
			PImage image = (PImage)raster.getRaster();
			image(image, 0, 0);
		}
		drawDetectors();
	}
	
	public void drawDetectors(){
		stroke(255);
		Iterator<Recipient> iter = recipients.iterator();
		while(iter.hasNext()){
			Recipient r = iter.next();
			try{
				ListIterator<Detector> i = r.getDetectorPatchList().listIterator();
				int channel = 0;
				while(i.hasNext()){
					Detector d = i.next();
					if (d != null){
						point(d.getX(), d.getY());
//						int val = face.getLastEvaluatedValue(d) & 0xFF;
//						int x = channel % faceWidth;
//						int y = channel / faceWidth;
//						stroke(val,0,0);
//						rect(x*12, y*24, 10, 10);
						//System.out.println(channel +" "+ val);
					}else{
						//System.out.println("- no detector -");
					}
					channel++;
				}
			} catch(NullPointerException e){
				e.printStackTrace();
			}
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
