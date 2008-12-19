package net.electroland.lafm.shows;

import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class Glockenspiel extends ShowThread {

	public Glockenspiel(DMXLightingFixture[] flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID) {
		super(flowers, soundManager, lifespan, fps, raster, ID);
	}

	@Override
	public void complete(PGraphics raster) {
		// TODO Auto-generated method stub

	}

	@Override
	public void doWork(PGraphics raster) {
		// TODO Auto-generated method stub

	}

}
