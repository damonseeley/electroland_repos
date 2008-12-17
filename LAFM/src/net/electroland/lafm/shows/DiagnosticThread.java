package net.electroland.lafm.shows;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PGraphics;

public class DiagnosticThread extends ShowThread implements SensorListener {

	boolean isOn = true;
	
	public DiagnosticThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flower, soundManager, lifespan, fps, raster);
	}

	public DiagnosticThread(DMXLightingFixture[] flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flowers, soundManager, lifespan, fps, raster);
	}

	@Override
	public void complete() {
		this.getRaster().background(0,0,0);		
	}

	@Override
	public void doWork() {
		int c = isOn ? 255 : 0;
		this.getRaster().background(c,c,c);
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		if (eventFixture == this.getFlowers()[0] && !isOn){
			this.forceStop();
		}
	}
}