package net.electroland.lafm.shows;

import java.util.List;
import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class PropellerThread extends ShowThread implements SensorListener{

	private int red, green, blue;
	private int originalred, originalgreen, originalblue;
	private float rotation, rotSpeed, acceleration, deceleration;
	private int fadeSpeed, topSpeed;
	private boolean speedUp, slowDown;
	private boolean startSound;
	private String soundFile;
	
	public PropellerThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.originalred = red;
		this.originalgreen = green;
		this.originalblue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 60;
	}
	
	public PropellerThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.originalred = red;
		this.originalgreen = green;
		this.originalblue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 60;
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
			super.playSound(soundFile);
			startSound = false;
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(128, 128);
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,256,256);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.fill(red, green, blue);
		raster.rect(0,0,300,40);
		raster.endDraw();
		if(rotation < topSpeed){
			rotation += rotSpeed;
		}
		if(speedUp){
			rotSpeed += acceleration;
		} else if(slowDown){
			rotSpeed -= deceleration;
			if(rotSpeed < 1){
				if(red > 1 || green > 1 || blue > 1){
					red = red - fadeSpeed;
					green = green - fadeSpeed;
					blue = blue - fadeSpeed;
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
			red = originalred;
			green = originalgreen;
			blue = originalblue;
		}
	}

}
