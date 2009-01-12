package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class WipeThread extends ShowThread implements SensorListener{
	
	private int red, green, blue, alpha, y, barWidth;
	private int wipeSpeed, fadeSpeed, age, duration;
	private String soundFile;
	private Properties physicalProps;
	private boolean startSound, fadeOut;

	public WipeThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, int wipeSpeed, int fadeSpeed,
			String soundFile, Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.wipeSpeed = wipeSpeed;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		alpha = 0;
		age = 0;
		y = raster.height;
		barWidth = 20;
		duration = (lifespan*fps) - (100/fadeSpeed);
		startSound = true;
	}
	
	public WipeThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, int wipeSpeed, int fadeSpeed,
			String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);

		this.red = red;
		this.green = green;
		this.blue = blue;
		this.wipeSpeed = wipeSpeed;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		alpha = 0;
		age = 0;
		y = raster.height;
		barWidth = 20;
		duration = (lifespan*fps) - (100/fadeSpeed);
		startSound = true;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		if(startSound){
			super.playSound(soundFile, physicalProps);
			startSound = false;
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.beginDraw();
		raster.fill(0,0,0,fadeSpeed);		// gradually overwrites with black
		raster.rect(0,0,256,256);
		raster.noStroke();
		raster.fill(red, green, blue);
		raster.rect(0,y,raster.width,barWidth);	// thin bar moves from bottom to top
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
			} else {
				cleanStop();
			}
		}
		age++;
		raster.endDraw();
		
		if(y >= 0 - barWidth){
			y -= wipeSpeed;
		} else {
			y = raster.height;
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			fadeOut = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			fadeOut = false;
			alpha = 0;
			age = 0;
		}
	}

}
