package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import org.apache.log4j.Logger;

import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

public class ImageSequenceThread extends ShowThread  implements SensorListener {

	static Logger logger = Logger.getLogger(ImageSequenceThread.class);
	
	private int index = 0;
	private PImage[] sequence;
	private boolean resize = true;
	private boolean isTinted = false;
	private int hue, brightness;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	
	public ImageSequenceThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, PImage[] sequence, boolean resize, String soundFile,
			Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
		if (sequence != null){			
			this.sequence = sequence;
		}else{
			logger.error("WARNING: IMAGE SEQUENCE WAS A NULL POINTER");
		}
		this.resize = resize;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}

	public ImageSequenceThread(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int priority, PImage[] sequence, boolean resize, String soundFile,
			Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, priority);
		if (sequence != null){			
			this.sequence = sequence;
		}else{
			logger.error("WARNING: IMAGE SEQUENCE WAS A NULL POINTER");
		}
		this.resize = resize;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}

	public void disableTint(){
		isTinted = false;
	}
	
	public void enableTint(int hue, int brightness) {
		isTinted = true;
		this.hue = hue;
		this.brightness = brightness;
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
		
		if(startSound){
			super.playSound(soundFile, physicalProps);
			startSound = false;
		}

		raster.beginDraw();

		if (isTinted){
			raster.colorMode(PConstants.HSB, 360, 100, 100);
			raster.tint(hue, 100, brightness);
		}
		
		if (resize){
			raster.image(sequence[index++], 0, 0, raster.width, raster.height);			
		}else{
			raster.image(sequence[index++], 0, 0);
		}

		raster.endDraw();

		if (index == sequence.length){
			//index = 0;	// make image sequences one-shot play back
			this.cleanStop();
		}
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		if (this.getFlowers().contains(eventFixture) && !isOn){
			//this.cleanStop();	// finish the animation even if someone leaves
		}
	}
}