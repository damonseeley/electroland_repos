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

public class SpinningRingThread extends ShowThread implements SensorListener{
	
	private int red, green, blue;						// color value
	private float brightness, fadeSpeed;				// brightness of color (for center throbbing)
	private float outerRot, innerRot;					// current rotational positions
	private float outerSpeed, innerSpeed, coreSpeed;	// brightness change increments
	private float acceleration, deceleration;			// subtract from hold durations
	private boolean speedUp, slowDown, fadeIn, fadeOut;
	private PImage outerRing, innerRing;
	private int alpha = 100;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;

	public SpinningRingThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int red, int green, int blue,
			float outerSpeed, float innerSpeed, float coreSpeed, float fadeSpeed,
			float acceleration, float deceleration, PImage outerRing, PImage innerRing,
			boolean startFast, String soundFile, Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.outerSpeed = outerSpeed;
		this.innerSpeed = innerSpeed;
		this.coreSpeed = coreSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.outerRing = outerRing;
		this.innerRing = innerRing;
		innerRot = 0;
		outerRot = 0;
		brightness = 255;
		if(startFast){
			speedUp = false;
			slowDown = true;
		} else {
			speedUp = true;
			slowDown = false;
		}
		fadeIn = true;
		fadeOut = false;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}
	
	public SpinningRingThread(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int red, int green, int blue,
			float outerSpeed, float innerSpeed, float coreSpeed, float fadeSpeed,
			float acceleration, float deceleration, PImage outerRing, PImage innerRing,
			boolean startFast, String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.outerSpeed = outerSpeed;
		this.innerSpeed = innerSpeed;
		this.coreSpeed = coreSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.outerRing = outerRing;
		this.innerRing = innerRing;
		innerRot = 0;
		outerRot = 0;
		brightness = 255;
		if(startFast){
			speedUp = false;
			slowDown = true;
		} else {
			speedUp = true;
			slowDown = false;
		}
		fadeIn = true;
		fadeOut = false;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
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
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.background(0);
		raster.translate(raster.width/2, raster.height/2);
		
		raster.pushMatrix();
		raster.rotate((float)(outerRot * Math.PI/180));
		raster.tint(red, green, blue, alpha);
		raster.image(outerRing, -raster.width/2, -raster.height/2, raster.width, raster.height);
		raster.popMatrix();
		
		raster.pushMatrix();
		raster.rotate((float)(innerRot * Math.PI/180));
		raster.tint(red, green, blue, alpha);
		raster.image(innerRing, -raster.width/4, -raster.width/4, raster.width/2, raster.height/2);
		raster.popMatrix();
		
		/*
		raster.fill((red/255.0f)*brightness, (green/255.0f)*brightness, (blue/255.0f)*brightness, alpha);
		raster.rect(0,0,raster.width/6,raster.height/6);	
		*/	
		raster.endDraw();
		
		if(fadeIn && brightness < 255){
			brightness += coreSpeed;
		} else if(fadeIn && brightness >= 255){
			brightness = 255;
			fadeIn = false;
			fadeOut = true;
		} else if(fadeOut && brightness > 0){
			brightness -= coreSpeed;
		} else if(fadeOut && brightness <= 0){
			brightness = 0;
			fadeIn = true;
			fadeOut = false;
		}
		
		outerRot += outerSpeed;
		innerRot += innerSpeed;
		if(speedUp){
			outerSpeed += acceleration;
			innerSpeed += acceleration;
			coreSpeed += acceleration*10;
		} else if(slowDown){
			if(outerSpeed > 0){
				outerSpeed -= deceleration;
			}
			if(innerSpeed > 0){
				innerSpeed -= deceleration;
			}
			if(coreSpeed > 0){
				coreSpeed -= deceleration*10;
			}
			if(outerSpeed < 0.001){
				/*
				if(alpha <= 0){
					cleanStop();					
				} else {
					alpha -= fadeSpeed;
				}
				*/
				cleanStop();
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
		}
	}

}
