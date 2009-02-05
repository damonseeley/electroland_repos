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

public class PropellerThread extends ShowThread implements SensorListener{

	private int red, green, blue, red2, green2, blue2;
	private float[] colorA, colorB;
	private float rotation, rotSpeed, acceleration, deceleration;
	private int fadeSpeed, topSpeed;
	private boolean speedUp, slowDown;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private float gain;
	int age = 0;
	private int duration;	// counting frames before fading out
	
	public PropellerThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile, Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		colorA = spectrum.getColor((float)Math.random());
		colorB = spectrum.getColor((float)Math.random());
		this.red = (int)colorA[0];
		this.green = (int)colorA[1];
		this.blue = (int)colorA[2];
		this.red2 = (int)colorB[0];
		this.green2 = (int)colorB[1];
		this.blue2 = (int)colorB[2];
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 20;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/fadeSpeed);
		this.physicalProps = physicalProps;
		this.gain = gain;
	}
	
	public PropellerThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			ColorScheme spectrum, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile, Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		colorA = spectrum.getColor((float)Math.random());
		colorB = spectrum.getColor((float)Math.random());
		this.red = (int)colorA[0];
		this.green = (int)colorA[1];
		this.blue = (int)colorA[2];
		this.red2 = (int)colorB[0];
		this.green2 = (int)colorB[1];
		this.blue2 = (int)colorB[2];
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 20;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/fadeSpeed);
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
			raster.beginDraw();
			raster.background(red,green,blue);
			raster.endDraw();
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(raster.width/2, raster.height/2);
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,raster.width,raster.height);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.fill(red, green, blue);
		raster.rect(0,0,raster.width + raster.width/5,raster.height/7);
		// A SECOND PROPELLER MIGHT BE A BAD IDEA
		raster.rotate((float)(90 * Math.PI/180));
		raster.fill(red2, green2, blue2);
		raster.rect(0,0,raster.width + raster.width/5,raster.height/7);
		raster.endDraw();
		
		if(age > duration){
			if(red > 1 || green > 1 || blue > 1){
				red = red - fadeSpeed;
				green = green - fadeSpeed;
				blue = blue - fadeSpeed;
				red2 -= fadeSpeed;
				green2 -= fadeSpeed;
				blue2 -= fadeSpeed;
			} else {
				cleanStop();
			}
		}
		age++;
		
		rotation += rotSpeed;
		if(speedUp){
			if(rotSpeed < topSpeed){
				rotSpeed += acceleration;
			}
		} else if(slowDown){
			if(rotSpeed <= 0){
				rotSpeed = 0;
			} else {
				rotSpeed -= deceleration;
			}
			if(rotSpeed < 1){
				if(red > 1 || green > 1 || blue > 1){
					red = red - fadeSpeed;
					green = green - fadeSpeed;
					blue = blue - fadeSpeed;
					red2 -= fadeSpeed;
					green2 -= fadeSpeed;
					blue2 -= fadeSpeed;
				} else {
					cleanStop();
				}
			}
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			//this.cleanStop();
			// potentially slow down when sensor triggered off
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
			red = (int)colorA[0];
			green = (int)colorA[0];
			blue = (int)colorA[0];
			red2 = (int)colorB[0];
			green2 = (int)colorB[1];
			blue2 = (int)colorB[2];
		}
	}

}
