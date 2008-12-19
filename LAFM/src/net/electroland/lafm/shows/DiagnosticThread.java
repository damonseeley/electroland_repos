package net.electroland.lafm.shows;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PGraphics;

public class DiagnosticThread extends ShowThread implements SensorListener {

	private static int WHITE = -1;
	private static int BLACK = -16777216;

	public DiagnosticThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flower, soundManager, lifespan, fps, raster);
	}

	public DiagnosticThread(DMXLightingFixture[] flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flowers, soundManager, lifespan, fps, raster);
	}

	@Override
	public void complete(PGraphics raster) {
		raster.background(DiagnosticThread.BLACK);	
	}

	@Override
	public void doWork(PGraphics raster) {
		raster.background(DiagnosticThread.WHITE);	
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (eventFixture == this.getFlowers()[0] && !isOn){
			this.cleanStop();
		}
	}
}