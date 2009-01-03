package net.electroland.lafm.shows;

import java.util.List;

import org.apache.log4j.Logger;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import processing.core.PConstants;
import processing.core.PGraphics;

public class ShutdownThread extends ShowThread {

	static Logger logger = Logger.getLogger(ShutdownThread.class);

	public ShutdownThread(List<DMXLightingFixture> flowers, PGraphics raster,
			String ID) {
		super(flowers, null, 0, 1, raster, ID, 1000000);
	}

	@Override
	public void doWork(PGraphics raster) {
		// won't get called, due to instantly short lifespan.
	}

	@Override
	public void complete(PGraphics raster) {
		logger.fatal("Sending black to everyone.");
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);	// paint it black.
		raster.endDraw();
	}
}