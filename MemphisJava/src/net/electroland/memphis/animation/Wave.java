package net.electroland.memphis.animation;

import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Wave implements Animation{
	
	private boolean isDone = false;
	private Raster raster;
	private Properties props;
	
	public Wave(PGraphics image){
		raster = new Raster(image);
		//System.out.println(raster.isProcessing());
	}

	public Raster getFrame() {
		PGraphics c = (PGraphics)raster.getRaster();
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		c.beginDraw();
		c.background(255);
		c.endDraw();
		return raster;
	}

	public void init(Properties props) {
		this.props = props;
	}

	public boolean isDone() {
		return isDone;
	}

	public void stop(){
		isDone = true;
	}

}
