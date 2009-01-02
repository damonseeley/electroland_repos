package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class SpinningRingThread extends ShowThread implements SensorListener{
	
	private float red, green, blue;						// normalized color value parameters
	private float brightness, fadeSpeed;				// brightness of color (for center throbbing)
	private float outerRot, innerRot;					// current rotational positions
	private float outerSpeed, innerSpeed, coreSpeed;	// brightness change increments
	private float acceleration, deceleration;			// subtract from hold durations
	private boolean speedUp, slowDown, fadeIn, fadeOut;

	public SpinningRingThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, int red, int green, int blue,
			float outerSpeed, float innerSpeed, float coreSpeed, float fadeSpeed,
			float acceleration, float deceleration) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = (float)(red/255.0);
		this.green = (float)(green/255.0);
		this.blue = (float)(blue/255.0);
		this.outerSpeed = outerSpeed;
		this.innerSpeed = innerSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		innerRot = 0;
		outerRot = 0;
		brightness = 255;
		speedUp = true;
		slowDown = false;
		fadeIn = true;
		fadeOut = false;
	}
	
	public SpinningRingThread(List <DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		// TODO Auto-generated constructor stub
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(128, 128);
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,256,256);
		raster.pushMatrix();
		raster.rotate((float)(outerRot * Math.PI/180));
		for(int i=0; i<18; i++){	// draw circle of values
			raster.pushMatrix();
			raster.rotate((float)((i*(360/18)) * Math.PI/180));
			if(i%2 == 0){
				raster.fill(red*255, green*255, blue*255);
			} else {
				raster.fill(0, 0, 0);
			}
			raster.rect(0, 100, 30, 50);
			raster.popMatrix();
		}
		raster.popMatrix();
		raster.pushMatrix();
		raster.rotate((float)(innerRot * Math.PI/180));
		//raster.fill(red*255, green*255, blue*255);
		//raster.rect(0, 50, 50, 50);
		for(int i=0; i<8; i++){	// draw circle of values
			raster.pushMatrix();
			raster.rotate((float)((i*(360/8)) * Math.PI/180));
			if(i%2 == 0){
				raster.fill(red*255, green*255, blue*255);
			} else {
				raster.fill(0, 0, 0);
			}
			raster.rect(0, 50, 50, 50);
			raster.popMatrix();
		}
		raster.popMatrix();
		raster.fill(red*brightness, green*brightness, blue*brightness);
		raster.rect(0, 0, 50, 50);
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
		} else if(slowDown){
			outerSpeed -= deceleration;
			innerSpeed -= deceleration;
			if(outerSpeed < 1){
				if(brightness > 1){
					brightness -= fadeSpeed;
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
		}
	}

}
