package net.electroland.lafm.shows;

import java.util.List;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PConstants;
import processing.core.PGraphics;

public class DiagnosticThread extends ShowThread implements SensorListener {

	private static int WHITE = 255;
	private static int BLACK = 0;

	public DiagnosticThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int priority) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
	}

	public DiagnosticThread(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int priority) {
		super(flowers, soundManager, lifespan, fps, raster, ID, priority);
	}

	@Override
	public void complete(PGraphics raster) {
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(DiagnosticThread.BLACK);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(DiagnosticThread.WHITE);
		raster.endDraw();
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			this.cleanStop();
		}
	}
}