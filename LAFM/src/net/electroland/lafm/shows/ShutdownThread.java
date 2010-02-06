package net.electroland.lafm.shows;

import java.util.List;

import org.apache.log4j.Logger;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PConstants;
import processing.core.PGraphics;

public class ShutdownThread extends ShowThread {

	static Logger logger = Logger.getLogger(ShutdownThread.class);

	public ShutdownThread(List<DMXLightingFixture> flowers, SoundManager soundManager, PGraphics raster,
			String ID) {
		super(flowers, soundManager, 5000, 30, raster, ID, 1000000);	// extended lifespan to make sure all fixtures are painted black
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
		logger.fatal("Shutting down SCSC.");
		super.getSoundManager().shutdown();
		logger.fatal("Sending black to everyone.");
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);	// paint it black.
		raster.endDraw();
	}
}