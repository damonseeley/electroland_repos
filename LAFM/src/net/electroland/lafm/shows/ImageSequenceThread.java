package net.electroland.lafm.shows;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

public class ImageSequenceThread extends ShowThread  implements SensorListener {

	private int index = 0;
	private PImage[] sequence;
	private boolean resize = true;
	
	public ImageSequenceThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, PImage[] sequence, boolean resize) {
		super(flower, soundManager, lifespan, fps, raster, ID);
		if (sequence != null){			
			this.sequence = sequence;
		}else{
			System.out.println("WARNING: IMAGE SEQUENCE WAS A NULL POINTER");
		}
		this.resize = resize;
	}

	public ImageSequenceThread(DMXLightingFixture[] flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, PImage[] sequence, boolean resize) {
		super(flower, soundManager, lifespan, fps, raster, ID);
		if (sequence != null){			
			this.sequence = sequence;
		}else{
			System.out.println("WARNING: IMAGE SEQUENCE WAS A NULL POINTER");
		}
		this.resize = resize;
	}

	@Override
	public void complete(PGraphics raster) {

		raster.colorMode(PConstants.RGB, 255, 255, 255);
		raster.beginDraw();
		raster.background(0);	// paint it black.
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		if (resize){
			raster.beginDraw();
			raster.image(sequence[index++], 0, 0, raster.width, raster.height);			
			raster.endDraw();
		}else{
			raster.beginDraw();
			raster.image(sequence[index++], 0, 0);
			raster.endDraw();
		}
		if (index == sequence.length){
			index = 0;
		}
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		if (eventFixture == this.getFlowers()[0] && !isOn){
			this.cleanStop();
		}		
	}
}
