package net.electroland.enteractive.shows;

import processing.core.PGraphics;
import net.electroland.animation.Animation;
import net.electroland.animation.Raster;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;

public class ExampleAnimation implements Animation {

	private Model m;
	private Raster r;
	private SoundManager sm;
	private int cycles = 30;
	
	public ExampleAnimation(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
	}
	
	public void initialize() {
		// play some sound, clear the raster, etc.
	}

	
	public Raster getFrame() {
		// lock the model so other people don't modify it while we do paint
		// calls based on it.
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics myRaster = (PGraphics)(r.getRaster());
		}
		return r;
	}

	public void cleanUp() {
		// play some sound, clear the raster, etc.
	}

	public boolean isDone() {
		return cycles-- > 0;
	}

}
