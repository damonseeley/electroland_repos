package net.electroland.enteractive.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.animation.Animation;
import net.electroland.animation.Raster;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;

public class ExampleAnimation implements Animation {

	private Model m;
	private Raster r;
	private SoundManager sm;
	private int cycles = 90;
	
	public ExampleAnimation(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		//System.out.println("instantiated.");
	}

	public void initialize() {
		// play some sound, clear the raster, etc.
		//System.out.println("initializing.");
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	
	public Raster getFrame() {
		// lock the model so other people don't modify it while we do paint
		// calls based on it.
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics myRaster = (PGraphics)(r.getRaster());
			myRaster.beginDraw();
			//myRaster.background(255,0,0); // fully on
			// VEGAS!
			for(int y=0; y<myRaster.height; y++){
				for(int x=0; x<myRaster.width; x++){
					myRaster.pixels[y*myRaster.width + x] = myRaster.color((int)(Math.random()*255),0,0);
				}
			}
			myRaster.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		// play some sound, clear the raster, etc.
		//System.out.println("cleaning up.");
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return cycles-- <= 0;
	}
}