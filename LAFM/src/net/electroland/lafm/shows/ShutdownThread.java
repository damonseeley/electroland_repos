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
		super(flowers, null, 5000, 30, raster, ID, 1000000);	// extended lifespan to make sure all fixtures are painted black
	}

	@Override
	public void doWork(PGraphics raster) {
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);	// paint it black.
		raster.endDraw();
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