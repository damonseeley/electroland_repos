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
	
	private int red, green, blue;							// color value
	private float brightness, fadeSpeed;					// brightness of color (for center throbbing)
	private float outerRot, innerRot;						// current rotational positions
	private float outerSpeed, innerSpeed, coreSpeed;		// brightness change increments
	private float innerAcceleration, innerDeceleration;	// subtract from hold durations
	private float outerAcceleration, outerDeceleration;	// subtract from hold durations
	private boolean speedUp, slowDown, fadeIn, fadeOut, fadeEverythingOut;
	private PImage outerRing, innerRing;
	private int alpha = 100;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int startDelay, delayCount;
	int age = 0;
	private int duration;	// counting frames before fading out
	private int topSpeed;
	private int outerRadius, innerRadius;

	public SpinningRingThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int red, int green, int blue,
			float outerSpeed, float innerSpeed, float coreSpeed, float fadeSpeed,
			float outerAcceleration, float outerDeceleration, float innerAcceleration,
			float innerDeceleration, PImage outerRing, PImage innerRing,
			boolean startFast, String soundFile, Properties physicalProps, int startDelay) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.outerSpeed = outerSpeed;
		this.innerSpeed = innerSpeed;
		this.coreSpeed = coreSpeed;
		this.fadeSpeed = fadeSpeed;
		this.innerAcceleration = innerAcceleration;
		this.innerDeceleration = innerDeceleration;
		this.outerAcceleration = outerAcceleration;
		this.outerDeceleration = outerDeceleration;
		this.outerRing = outerRing;
		this.innerRing = innerRing;
		
		if(outerSpeed < 0){											// mirror flip image
			PImage newOuterRing = new PImage(outerRing.width, outerRing.height, PConstants.ARGB);
			newOuterRing.loadPixels();
		    for (int i = 0; i < outerRing.width; i++) {			// Begin loop for width
		    	for (int j = 0; j < outerRing.height; j++) {    	// Begin loop for height  
		    		newOuterRing.pixels[j*outerRing.width+i] = outerRing.pixels[(outerRing.width - i - 1) + j*outerRing.width];
		    	}
		    } 
		    newOuterRing.updatePixels();
		    this.outerRing = newOuterRing;
		}
		if(innerSpeed < 0){	// mirror flip image
			PImage newInnerRing = new PImage(innerRing.width, innerRing.height, PConstants.ARGB);
			newInnerRing.loadPixels();
		    for (int i = 0; i < innerRing.width; i++) {			// Begin loop for width
		    	for (int j = 0; j < innerRing.height; j++) {    	// Begin loop for height  
		    		newInnerRing.pixels[j*innerRing.width+i] = innerRing.pixels[(innerRing.width - i - 1) + j*innerRing.width];
		    	}
		    } 
		    newInnerRing.updatePixels();
		    this.innerRing = newInnerRing;
		}
		
		outerRadius = raster.width/2;
		innerRadius = raster.width/4;
		//this.outerRing.resize(outerRadius*2, outerRadius*2);	// getting poor results from this
		//this.innerRing.resize(innerRadius*2, innerRadius*2);
		
		duration = (lifespan*fps) - (int)(100/fadeSpeed);
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
		fadeEverythingOut = false;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		delayCount = 0;
		topSpeed = 45;
		startSound = true;
	}
	
	public SpinningRingThread(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int red, int green, int blue,
			float outerSpeed, float innerSpeed, float coreSpeed, float fadeSpeed,
			float outerAcceleration, float outerDeceleration, float innerAcceleration,
			float innerDeceleration, PImage outerRing, PImage innerRing,
			boolean startFast, String soundFile, Properties physicalProps, int startDelay) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.outerSpeed = outerSpeed;
		this.innerSpeed = innerSpeed;
		this.coreSpeed = coreSpeed;
		this.fadeSpeed = fadeSpeed;
		this.innerAcceleration = innerAcceleration;
		this.innerDeceleration = innerDeceleration;
		this.outerAcceleration = outerAcceleration;
		this.outerDeceleration = outerDeceleration;
		this.outerRing = outerRing;
		this.innerRing = innerRing;
		outerRadius = raster.width/2;
		innerRadius = raster.width/4;
		//this.outerRing.resize(outerRadius*2, outerRadius*2);
		//this.innerRing.resize(innerRadius*2, innerRadius*2);
		duration = (lifespan*fps) - (int)(100/fadeSpeed);
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
		fadeEverythingOut = false;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		delayCount = 0;
		topSpeed = 45;
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
			raster.rectMode(PConstants.CENTER);
			raster.beginDraw();
			raster.noStroke();
			raster.background(0);
			raster.translate(raster.width/2, raster.height/2);
			
			raster.pushMatrix();
			raster.rotate((float)(outerRot * Math.PI/180));
			raster.tint(red, green, blue, alpha);
			raster.image(outerRing, -raster.width/2, -raster.height/2, raster.width, raster.height);
			//raster.image(outerRing, -outerRadius, -outerRadius);
			raster.popMatrix();
			
			raster.pushMatrix();
			raster.rotate((float)(innerRot * Math.PI/180));
			raster.tint(red, green, blue, alpha);
			raster.image(innerRing, -raster.width/4, -raster.width/4, raster.width/2, raster.height/2);
			//raster.image(innerRing, -innerRadius, -innerRadius);
			raster.popMatrix();
			
			if(age > duration){
				fadeEverythingOut = true;
			}
			
			if(fadeEverythingOut){
				if(alpha > 0){
					alpha -= fadeSpeed;
				} 
				if(alpha <= 0){
					cleanStop();
				}
			}
			//age++;
			
			raster.endDraw();
			/*
			// this was originally for the core, which is no longer in use
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
			*/
			
			outerRot += outerSpeed;
			innerRot += innerSpeed;
			if(speedUp){
				if(Math.abs(outerSpeed) < topSpeed){
					outerSpeed += outerAcceleration;
				}
				if(Math.abs(innerSpeed) < topSpeed){
					innerSpeed += innerAcceleration;
				}
				coreSpeed += innerAcceleration*10;
			} else if(slowDown){
				if(Math.abs(outerSpeed) > 0){
					outerSpeed -= outerDeceleration;
				}
				if(Math.abs(innerSpeed) > 0){
					innerSpeed -= innerDeceleration;
				}
				if(coreSpeed > 0){
					coreSpeed -= innerDeceleration*10;
				}
				if(Math.abs(outerSpeed) < 1){
					fadeEverythingOut = true;
					/*
					if(alpha <= 0){
						cleanStop();					
					} else {
						alpha -= fadeSpeed;
					}
					*/
				}
			}
		} else {
			delayCount++;
			//super.resetLifespan();
		}
		age++;
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
			fadeEverythingOut = false;
			speedUp = true;
			slowDown = false;
			alpha = 100;
		}
	}

}
