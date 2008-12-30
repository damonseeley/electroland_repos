package net.electroland.lafm.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ExpandingThread extends ShowThread {
	
	PImage texture;
	Expander[] expanders;

	public ExpandingThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, PImage texture) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.texture = texture;
		expanders = new Expander[3];
		expanders[0] = new Expander(255,0,0,200);
		expanders[1] = new Expander(0,255,0,125);
		expanders[2] = new Expander(0,0,255,50);
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		raster.beginDraw();
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.background(0);
		raster.translate(128,128);
		for(int i=0; i<expanders.length; i++){
			expanders[i].draw(raster);
		}
		raster.endDraw();
	}
	
	public void changeOrder(){
		Expander[] neworder = new Expander[expanders.length];
		System.arraycopy(expanders, 1, neworder, 0, expanders.length-1);
		neworder[expanders.length-1] = expanders[0];	// makes oldest the newest
		expanders = neworder;
	}
	
	
	
	
	private class Expander{
		private int red, green, blue;
		private int diameter;
		
		private Expander(int red, int green, int blue, int diameter){
			this.red = red;
			this.green = green;
			this.blue = blue;
			this.diameter = diameter;
		}
		
		private void draw(PGraphics raster){
			if(diameter < 400){
				diameter += 5;
			} else {
				// destroy this and create a new tiny one
				// OR resort the array order so this is drawn last
				diameter = 10;
				changeOrder();
			}
			raster.tint(red,green,blue);
			raster.image(texture,0-diameter/2,0-diameter/2,diameter,diameter);
			//raster.fill(red,green,blue);
			//raster.ellipse(0,0,diameter,diameter);
		}
	}

}
