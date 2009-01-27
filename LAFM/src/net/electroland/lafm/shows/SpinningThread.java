package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class SpinningThread extends ShowThread implements SensorListener{

	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int startDelay, delayCount;
	private PImage sprite;
	private int rotation;
	private float rotSpeed, acceleration, deceleration;
	private boolean speedUp, slowDown, fadeOut;
	private int age = 0;
	private int alpha = 100;
	private int duration;	// counting frames before fading out
	private int topSpeed, fadeSpeed;
	
	public SpinningThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			PImage sprite, float rotSpeed, float acceleration, float deceleration,
			int fadeSpeed, String soundFile, Properties physicalProps, int startDelay) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.sprite = sprite;
		this.rotSpeed = rotSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		rotation = 0;
		delayCount = 0;
		duration = (lifespan*fps) - (int)(100/fadeSpeed);
		topSpeed = 45;
		fadeOut = false;
		speedUp = true;
		slowDown = false;
		startSound = true;
	}
	
	public SpinningThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			PImage sprite, float rotSpeed, float acceleration, float deceleration,
			int fadeSpeed, String soundFile, Properties physicalProps, int startDelay) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.sprite = sprite;
		this.rotSpeed = rotSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		rotation = 0;
		delayCount = 0;
		duration = (lifespan*fps) - (int)(100/fadeSpeed);
		topSpeed = 45;
		fadeOut = false;
		speedUp = true;
		slowDown = false;
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
		if(delayCount >= startDelay){
			if(startSound){
				super.playSound(soundFile, physicalProps);
				startSound = false;
			}
			
			raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
			raster.beginDraw();
			raster.noStroke();
			raster.background(0);
			raster.translate(raster.width/2, raster.height/2);
			raster.rotate((float)(rotation * Math.PI/180));
			raster.tint(255, 255, 255, alpha);	// default full color
			raster.image(sprite,-raster.width/2,-raster.height/2,raster.width,raster.height);
			
			if(age > duration){
				fadeOut = true;
			}
			
			if(fadeOut){
				if(alpha > 0){
					alpha -= fadeSpeed;
				} 
				if(alpha <= 0){
					cleanStop();
				}
			}
			age++;
			
			raster.endDraw();
			
			rotation += rotSpeed;
			if(speedUp){
				if(Math.abs(rotSpeed) < topSpeed){
					rotSpeed += acceleration;
				}
			} else if(slowDown){
				if(Math.abs(rotSpeed) > 0){
					rotSpeed -= deceleration;
				}
				if(Math.abs(rotSpeed) < 1){
					if(alpha <= 0){
						cleanStop();					
					} else {
						alpha -= fadeSpeed;
					}
				}
			}
		} else {
			delayCount++;
			super.resetLifespan();
		}
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		if (this.getFlowers().contains(eventFixture) && !isOn){
			// slow down when sensor triggered off
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
		}
	}
	
}
