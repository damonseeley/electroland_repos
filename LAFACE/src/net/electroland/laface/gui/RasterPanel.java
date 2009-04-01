package net.electroland.laface.gui;

import java.awt.event.ActionEvent;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

import net.electroland.laface.core.LAFACEMain;
import net.electroland.lighting.detector.Detector;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Raster;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

/**
 * Displays the raster image as well as allows toggling of lights directly.
 * @author asiegel
 */

@SuppressWarnings("serial")
public class RasterPanel extends PApplet {
	
	private LAFACEMain main;
	private Raster raster;
	private Collection<Recipient> recipients;
	private int guiWidth, guiHeight;
	private int faceWidth, faceHeight, lightWidth, lightHeight;		// light grid
	private List<Light> lights;
	
	public RasterPanel(LAFACEMain main, Collection<Recipient> recipients, int faceWidth, int faceHeight){
		this.main = main;
		this.recipients = recipients;
		this.faceWidth = faceWidth;
		this.faceHeight = faceHeight;
		lights = new ArrayList<Light>();
		guiWidth = 1048;
		guiHeight = 133;
	}
	
	public void setRaster(Raster raster){
		this.raster = raster;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			lightWidth = c.width/faceWidth;
			lightHeight = c.height/faceHeight;
		}
		lights.clear();
		for(int i=0; i<faceWidth*faceHeight; i++){
			lights.add(new Light(i, i%faceWidth, i/faceWidth, lightWidth, lightHeight));
		}
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
		if(lights.size() > 0){
			int x = (mouseX / lightWidth);		// grid location
			int y = (mouseY / lightHeight);
			int id = x + y*faceWidth;
			if(id < lights.size()){
				if(mouseButton == LEFT){
					if(keyPressed && keyCode == SHIFT){
						lights.get(id).turnOff();
					} else {
						lights.get(id).turnOn();
					}
				} else if(mouseButton == RIGHT){
					lights.get(id).turnOff();
				}
			}
		}
	}
	
	public void mousePressed(){
		if(lights.size() > 0){
			int x = (mouseX / lightWidth);		// grid location
			int y = (mouseY / lightHeight);
			int id = x + y*faceWidth;
			if(id < lights.size()){
				if(mouseButton == LEFT){
					if(keyPressed && keyCode == SHIFT){
						lights.get(id).turnOff();
					} else {
						lights.get(id).turnOn();
					}
				} else if(mouseButton == RIGHT){
					lights.get(id).turnOff();
				}
			}
		}
	}
	
	
	
	
	
	private class Light{
		
		private int id, x, y, width, height, brightness;
		
		private Light(int id, int x, int y, int width, int height){
			this.id = id;
			this.x = x;
			this.y = y;
			this.width = width;
			this.height = height;
			brightness = 0;
		}
		
		public void turnOn(){
			brightness = 255;
			ActionEvent event = new ActionEvent(this, ActionEvent.ACTION_PERFORMED, "turnOn:"+String.valueOf(id));
			main.actionPerformed(event);
		}
		
		public void turnOff(){
			brightness = 0;
			ActionEvent event = new ActionEvent(this, ActionEvent.ACTION_PERFORMED, "turnOff:"+String.valueOf(id));
			main.actionPerformed(event);
		}
	}

}
