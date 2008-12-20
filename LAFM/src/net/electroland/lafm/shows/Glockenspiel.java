package net.electroland.lafm.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class Glockenspiel extends ShowThread {
	
	private int hour, minute, sec;

	public Glockenspiel(DMXLightingFixture[] flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int hour, int minute, int sec) {
		super(flowers, soundManager, lifespan, fps, raster, ID);
		this.hour = hour;
		this.minute = minute;
		this.sec = sec;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.background(0);

	}

	@Override
	public void doWork(PGraphics raster) {
		raster.colorMode(PConstants.HSB, 60, 255, 255);
		raster.beginDraw();
		raster.background(this.minute, 255, 255);
		raster.endDraw();
	}

}
