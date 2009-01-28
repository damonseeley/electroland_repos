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

public class DartBoardThread extends ShowThread implements SensorListener{
	
	ColorScheme spectrum;
	float val1, val2, val3, offset;
	float speed, acceleration, deceleration;
	float[] color;
	boolean speedUp, slowDown;
	private boolean startSound, fadeOut;
	private String soundFile;
	private Properties physicalProps;
	private float topSpeed;
	int age = 0;
	int alpha = 0;
	int fadeSpeed = 3;
	private int duration;	// counting frames before fading out
	private boolean modulate;

	public DartBoardThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float offset, float acceleration, float deceleration, String soundFile,
			Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.offset = offset;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		val1 = 0;
		val2 = val1 + offset;
		val3 = val2 + offset;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		topSpeed = 0.15f;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
		modulate = false;
	}
	
	public DartBoardThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float offset, float acceleration, float deceleration, String soundFile,
			Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.offset = offset;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		val1 = 0;
		val2 = val1 + offset;
		val3 = val2 + offset;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = false;		// sound is turned off to be triggered as a global sound
		this.physicalProps = physicalProps;
		topSpeed = 0.15f;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
		modulate = true;
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
		raster.ellipseMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(raster.width/2, raster.height/2);
		float[] colora = spectrum.getColor(val1);
		raster.fill(colora[0],colora[1],colora[2]);
		raster.ellipse(0,0,raster.width,raster.height);
		float[] colorb = spectrum.getColor(val2);
		raster.fill(colorb[0],colorb[1],colorb[2]);
		raster.ellipse(0,0,(raster.width/5)*3,(raster.height/5)*3);
		float[] colorc = spectrum.getColor(val3);
		raster.fill(colorc[0],colorc[1],colorc[2]);
		raster.ellipse(0,0,raster.width/5,raster.height/5);
		
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(-(raster.width/2),-(raster.height/2),raster.width,raster.height);
			} 
			if(alpha >= 100){
				raster.fill(0,0,0,alpha);
				raster.rect(-(raster.width/2),-(raster.height/2),raster.width,raster.height);
				cleanStop();
			}
		}
		age++;
		
		raster.endDraw();
		
		if(val1 >= 1){
			val1 = val1 - 1;
		} else {
			val1 += speed;
		}
		if(val2 >= 1){
			val2 = val2 - 1;
		} else {
			val2 += speed;
		}
		if(val3 >= 1){
			val3 = val3 - 1;
		} else {
			val3 += speed;
		}
		
		if(modulate){	// this is show mode
			if(speedUp){
				if(speed < topSpeed){
					speed += acceleration;
				} else {
					speedUp = false;
					slowDown = true;
				}
			} else if(slowDown){
				if(speed > 0.005){
					speed -= deceleration;
				} else {
					slowDown = false;
					speedUp = true;
				}
			}
		} else {		// this is sensor mode
			if(speedUp){
				if(speed < topSpeed){
					speed += acceleration;
				}
			} else if(slowDown){
				if(speed > 0){
					speed -= deceleration;
				}
				if(speed < 0.01){
					fadeOut = true;
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
			fadeOut = false;
			alpha = 0;
			age = 0;
		}
	}

}
