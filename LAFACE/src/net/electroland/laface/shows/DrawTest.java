package net.electroland.laface.shows;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class DrawTest implements Animation{
	
	private Raster r;
	private int gridWidth, gridHeight, lightWidth, lightHeight;
	private List<Light> lights;
	
	public DrawTest(Raster r, int gridWidth, int gridHeight){
		this.r = r;
		this.gridWidth = gridWidth;
		this.gridHeight = gridHeight;
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			lightWidth = c.width/gridWidth;
			lightHeight = c.height/gridHeight;
		}
		lights = new ArrayList<Light>();
		for(int i=0; i<gridWidth*gridHeight; i++){
			lights.add(new Light(i, i%gridWidth, i/gridWidth, lightWidth, lightHeight));
		}
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		c.rectMode(PConstants.CENTER);
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			c.noStroke();
			Iterator<Light> iter = lights.iterator();
			while(iter.hasNext()){
				Light light = iter.next();
				c.fill(light.getValue());
				c.rect(light.x*lightWidth, light.y*lightHeight, light.width, light.height);
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
	
	public void turnOn(int id){
		Light light = lights.get(id);
		light.turnOn();
	}
	
	public void turnOff(int id){
		Light light = lights.get(id);
		light.turnOff();
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
			if((x+1) % 5 == 0){
				brightness = 0;
			}
		}
		
		public void turnOn(){
			if((x+1) % 5 != 0){
				brightness = 255;
			}
		}
		
		public void turnOff(){
			if((x+1) % 5 != 0){
				brightness = 0;
			}
		}
		
		public int getValue(){
			return brightness;
		}
	}

}
