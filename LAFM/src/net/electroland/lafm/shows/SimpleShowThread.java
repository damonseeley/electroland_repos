package net.electroland.lafm.shows;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PGraphics;

public class SimpleShowThread extends ShowThread implements SensorListener {

	public SimpleShowThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flower, soundManager, lifespan, fps, raster);
		
		// any other start up customization.  sound, config, etc.
	}
	
	
	public SimpleShowThread(DMXLightingFixture[] flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster) {
		super(flowers, soundManager, lifespan, fps, raster);
		
		// any other start up customization.  sound, config, etc.
	}


	@Override
	public void doWork(PGraphics raster) {

		// draw on the raster here.
		
	}

	
	@Override
	public void complete(PGraphics raster) {
		// if you want to do any tear down, do it here.  there's no frame
		// renders left, so this is mostly useful for a sound etc.
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {

		this.getFlowers(); // the fixtures you are watching.
		
		// if a midi even occurs while you are running, you'll be notified here.
		// you should check to see if the midi even is from a fixture that
		// you are watching. 
	}
}