package net.electroland.lafm.shows;

import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ImageSequenceThread extends ShowThread {

	// constructor should take an array of Images.
	
	public ImageSequenceThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID) {
		super(flower, soundManager, lifespan, fps, raster, ID);
		// TODO find the appropriate image
	}

	@Override
	public void complete(PGraphics raster) {
		// TODO Auto-generated method stub
		
		// paint black.
		
	}

	@Override
	public void doWork(PGraphics raster) {
		// TODO Auto-generated method stub

		// paint the image on the raster
		
	}

}
