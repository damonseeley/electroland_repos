package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ThrobbingThread extends ShowThread implements SensorListener{

	private float red, green, blue;					// normalized color value parameters
	private float brightness;							// intensity changed in throbbing
	private float holdon, holdoff;					// timing parameters in frames
	private int holdcount;
	private float fadeinspeed, fadeoutspeed;			// brightness change increments
	private int acceleration, deceleration;			// subtract from hold durations
	private boolean speedUp, slowDown;
	private int state;									// 0 = fade in, 1 = hold on, 2 = fade out, 3 = hold off
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;

	public ThrobbingThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int priority,
			int red, int green, int blue, int fadein, int fadeout, int holdon,
			int holdoff, int acceleration, int deceleration, String soundFile,
			Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, priority);
		this.red = (float)(red/255.0);
		this.green = (float)(green/255.0);
		this.blue = (float)(blue/255.0);
		if(fadein == 0){
			this.fadeinspeed = 255;
		} else {
			this.fadeinspeed = 255 / ((float)(fadein/1000.0)*fps);
		}
		if(fadeout == 0){
			this.fadeoutspeed = 255;
		} else {
			this.fadeoutspeed = 255 / ((float)(fadeout/1000.0)*fps);
		}
		if(this.holdon > 0){
			this.holdon = ((float)(holdon/1000.0)*fps);
		} else {
			this.holdon = 0;
		}
		if(this.holdoff > 0){
			this.holdoff = ((float)(holdoff/1000.0)*fps);
		} else {
			this.holdoff = 0;
		}
		holdcount = 0;
		this.brightness = 255;
		this.state = 1;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		raster.colorMode(PConstants.RGB, 255, 255, 255);
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}
	
	public ThrobbingThread(List <DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int priority,
			int red, int green, int blue, int fadein, int fadeout, int holdon,
			int holdoff, int acceleration, int deceleration, String soundFile,
			Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, priority);
		this.red = (float)(red/255.0);
		this.green = (float)(green/255.0);
		this.blue = (float)(blue/255.0);
		if(fadein == 0){
			this.fadeinspeed = 255;
		} else {
			this.fadeinspeed = 255 / ((float)(fadein/1000.0)*fps);
		}
		if(fadeout == 0){
			this.fadeoutspeed = 255;
		} else {
			this.fadeoutspeed = 255 / ((float)(fadeout/1000.0)*fps);
		}
		if(this.holdon > 0){
			this.holdon = ((float)(holdon/1000.0)*fps);
		} else {
			this.holdon = 0;
		}
		if(this.holdoff > 0){
			this.holdoff = ((float)(holdoff/1000.0)*fps);
		} else {
			this.holdoff = 0;
		}
		holdcount = 0;
		this.brightness = 255;
		this.state = 1;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		speedUp = true;
		slowDown = false;
		raster.colorMode(PConstants.RGB, 255, 255, 255);
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
		//System.out.println(this.brightness);
		
		if(startSound){
			super.playSound(soundFile, physicalProps);
			startSound = false;
		}
		
		if(state == 0){				// fade in
			if(brightness < 255){
				brightness += fadeinspeed;
			}
			if(brightness >= 255){
				state = 1;
				holdcount = 0;
			}
		} else if(state == 2){		// fade out
			if(brightness > 0){
				brightness -= fadeoutspeed;
			}
			if(brightness <= 0){
				state = 3;
			}
		}
		
		if(state == 1){				// hold on
			if(holdcount < holdon){
				holdcount++;
			} else {
				state = 2;
				holdcount = 0;
			}
		} else if(state == 3){		// hold off
			if(holdcount < holdoff){
				holdcount++;
			} else {
				state = 0;
				holdcount = 0;
			}
		}
		
		if(speedUp){
			if(fadeinspeed < 255){
				fadeinspeed += acceleration;
			}
			if(fadeoutspeed < 255){
				fadeoutspeed += acceleration;
			}
		} else if(slowDown){
			if(fadeinspeed > 1){
				fadeinspeed -= deceleration;
			} else {
				if(brightness > 1){
					brightness -= 5;
				} else {
					cleanStop();
				}
			}
			if(fadeoutspeed > 1){
				fadeoutspeed -= deceleration;
			} else {
				if(brightness > 1){
					brightness -= 5;
				} else {
					cleanStop();
				}
			}
		}
		
		/*
		if(speedUp){
			if(holdon > 0){
				holdon -= acceleration;
			}
			if(holdoff > 0){
				holdoff -= acceleration;
			}
		} else if(slowDown){
			holdon += deceleration;
			holdoff += deceleration;
			if(deceleration > 0){
				if(holdoff > 1000){
					cleanStop();
				}
			} else {
				cleanStop();
			}
		}
		*/
		
		raster.beginDraw();
		raster.background(red*brightness, green*brightness, blue*brightness);
		raster.endDraw();
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
		}
	}
}
