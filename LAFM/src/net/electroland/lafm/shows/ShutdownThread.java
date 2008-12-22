package net.electroland.lafm.shows;

import java.util.Collection;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ShutdownThread extends ShowThread {


	public ShutdownThread(Collection<DMXLightingFixture> flowers, PGraphics raster,
			String ID) {
		super(flowers, null, 0, 1, raster, ID, 1000000);
	}

	@Override
	public void doWork(PGraphics raster) {
		// won't get called, due to instantly short lifespan.
	}

	@Override
	public void complete(PGraphics raster) {
		System.out.println("Sending black to everyone.");
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);	// paint it black.
		raster.endDraw();
	}
}