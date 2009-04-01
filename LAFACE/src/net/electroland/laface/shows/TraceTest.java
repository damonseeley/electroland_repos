package net.electroland.laface.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class TraceTest implements Animation {
	
	private Raster r;
	private int gridWidth, gridHeight, lightWidth, lightHeight, tracerPos, speed;
	private long startTime;

	public TraceTest(Raster r, int gridWidth, int gridHeight, int speed){
		this.r = r;
		this.gridWidth = gridWidth;
		this.gridHeight = gridHeight;
		this.speed = speed;
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			lightWidth = c.width/gridWidth;
			lightHeight = c.height/gridHeight;
		}
		tracerPos = 0;
		startTime = System.currentTimeMillis();
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
			if(System.currentTimeMillis() - startTime > speed){
				if(tracerPos < gridWidth*gridHeight){
					tracerPos++;
				} else {
					tracerPos = 0;
				}
				startTime = System.currentTimeMillis();
			}
			c.fill(255,255,255,255);
			c.noStroke();
			int x = tracerPos % gridWidth;
			int y = tracerPos / gridWidth;
			if((x+1) % 5 == 0){
				x++;
				tracerPos++;
			}
			c.rect(x*lightWidth, y*lightHeight, lightWidth, lightHeight);
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
