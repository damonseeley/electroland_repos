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
	private int spriteSize;
	private boolean interactive;
	private float gain;
	
	public SpinningThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			PImage sprite, float rotSpeed, float acceleration, float deceleration,
			int fadeSpeed, String soundFile, Properties physicalProps, int startDelay,
			boolean interactive, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.sprite = sprite;
		this.rotSpeed = rotSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		this.interactive = interactive;
		this.gain = gain;
		rotation = 0;
		delayCount = 0;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/fadeSpeed);
		topSpeed = 45;
		fadeOut = false;
		speedUp = true;
		slowDown = false;
		startSound = true;
		spriteSize = raster.width;
		//this.sprite.resize(spriteSize, spriteSize);
	}
	
	public SpinningThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			PImage sprite, float rotSpeed, float acceleration, float deceleration,
			int fadeSpeed, String soundFile, Properties physicalProps, int startDelay,
			boolean interactive, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		
		this.sprite = sprite;
		this.rotSpeed = rotSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.fadeSpeed = fadeSpeed;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		this.interactive = interactive;
		this.gain = gain;
		rotation = 0;
		delayCount = 0;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/fadeSpeed);
		topSpeed = 45;
		fadeOut = false;
		speedUp = true;
		slowDown = false;
		startSound = true;
		spriteSize = raster.width;
		//this.sprite.resize(spriteSize, spriteSize);
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
				super.playSound(soundFile, gain, physicalProps);
				startSound = false;
			}
			
			raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
			raster.beginDraw();
			raster.noStroke();
			raster.background(0);
			raster.translate(raster.width/2, raster.height/2);
			raster.rotate((float)(rotation * Math.PI/180));
			raster.tint(255, 255, 255, alpha);	// default full color
			//raster.image(sprite,-raster.width/2,-raster.height/2,raster.width,raster.height);
			raster.image(sprite, -spriteSize/2, -spriteSize/2);
			
			//System.out.println(age +" "+ duration);
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
			//age++;
			
			raster.endDraw();
			
			rotation += rotSpeed;
			if(speedUp){
				if(Math.abs(rotSpeed) < topSpeed){
					rotSpeed += acceleration;
				}
			} else if(slowDown && age > 30){
				if(rotSpeed > 0 && deceleration > 0){
					rotSpeed -= deceleration;
					if(rotSpeed <= 0){
						rotSpeed = 0;
					}
				} else if(rotSpeed < 0 && deceleration < 0){
					rotSpeed -= deceleration;
					if(rotSpeed >= 0){
						rotSpeed = 0;
					}
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
			// all of the shows start at the same point, and thus will end at the same point
			// they only appear to start at different points
			//super.resetLifespan();
		}
		age++;
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		if(interactive){
			if (this.getFlowers().contains(eventFixture) && !isOn){
				// slow down when sensor triggered off
				speedUp = false;
				slowDown = true;
				deceleration = rotSpeed / 60;	// deceleration based on current speed,
			} else if(this.getFlowers().contains(eventFixture) && isOn){
				// reactivate
				speedUp = true;
				slowDown = false;
				alpha = 100;
			}
		}
	}
	
}
