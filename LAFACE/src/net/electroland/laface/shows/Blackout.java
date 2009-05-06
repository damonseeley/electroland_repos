package net.electroland.laface.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.lighting.detector.Recipient;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

/**
 * Just used to clear the raster before killing the app or recipient
 * @author Aaron Siegel
 */

public class Blackout implements Animation {

	private Raster r;
	private int duration;
	private Recipient recipient;
	private boolean toggleDetectors;
	private long startTime;
	
	public Blackout(Raster r, int duration, Recipient recipient, boolean toggleDetectors){
		this.r = r;
		this.duration = duration;
		this.recipient = recipient;
		this.toggleDetectors = toggleDetectors;
		startTime = System.currentTimeMillis();
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.beginDraw();
		raster.background(0);		// clear the raster
		raster.endDraw();
		return r;
	}

	public boolean isDone() {
		if(System.currentTimeMillis() - startTime > duration){
			if(toggleDetectors){
				//recipient.toggleDetectors();
			}
			return true;
		}
		return false;
	}

}
