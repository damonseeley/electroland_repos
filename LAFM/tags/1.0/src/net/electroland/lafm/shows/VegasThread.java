package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class VegasThread extends ShowThread implements SensorListener{
	
	ColorScheme spectrum;
	float speed;
	int age = 0;
	int delay = 0;
	int alpha = 0;
	int fadeSpeed = 3;
	private boolean startSound, fadeOut;
	private String soundFile;
	private int duration;	// counting frames before fading out
	private Properties physicalProps;
	private float minColorPoint, maxColorPoint;
	private float spectrumShiftSpeed;
	private boolean spectrumDirection;
	private float gain;

	public VegasThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, float minColorPoint, float maxColorPoint,
			float spectrumShiftSpeed, String soundFile, Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.minColorPoint = minColorPoint;
		this.maxColorPoint = maxColorPoint;
		this.spectrumShiftSpeed = spectrumShiftSpeed;
		this.soundFile = soundFile;
		this.spectrumDirection = true;
		startSound = true;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
		this.physicalProps = physicalProps;
		this.gain = gain;
	}
	
	public VegasThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float speed, float minColorPoint, float maxColorPoint,
			float spectrumShiftSpeed, String soundFile, Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.minColorPoint = minColorPoint;
		this.maxColorPoint = maxColorPoint;
		this.spectrumShiftSpeed = spectrumShiftSpeed;
		this.soundFile = soundFile;
		this.spectrumDirection = true;
		startSound = true;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
		this.physicalProps = physicalProps;
		this.gain = gain;
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
			super.playSound(soundFile, gain, physicalProps);
			startSound = false;
		}
		
		if(maxColorPoint >= 1){						// color range moving up or down spectrum
			spectrumDirection = false;
			maxColorPoint = 1;
		} else if(minColorPoint <= 0){
			spectrumDirection = true;
			minColorPoint = 0;
		}
		
		if(spectrumDirection){						// color range movement
			maxColorPoint += spectrumShiftSpeed;
			minColorPoint += spectrumShiftSpeed;
		} else {
			maxColorPoint -= spectrumShiftSpeed;
			minColorPoint -= spectrumShiftSpeed;
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		if(delay > speed){
			raster.background(0);
			raster.noStroke();
			for(int i=0; i<25; i++){
				float[] color = spectrum.getColor((float)(Math.random()*(maxColorPoint-minColorPoint))+minColorPoint);
				raster.fill(color[0], color[1], color[2]);
				raster.rect((float)(Math.random()*(raster.width-1)), (float)(Math.random()*(raster.height-1)), raster.width/5, raster.height/5);
			}
			delay = 0;
		} else {
			delay++;
		}
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut && age > 30){
			if(alpha < 100){
				alpha += fadeSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(raster.width/2,raster.height/2,raster.width,raster.height);
			} 
			if(alpha >= 100){
				raster.fill(0,0,0,alpha);
				raster.rect(-(raster.width/2),-(raster.height/2),raster.width,raster.height);
				cleanStop();
			}
		}
		raster.endDraw();
		age++;
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			// slow down when sensor triggered off
			fadeOut = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			fadeOut = false;
			alpha = 0;
			age = 0;
		}
	}

}
