package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class AdditivePropellerThread extends ShowThread implements SensorListener{
	
	private int red, green, blue;
	private float rotation, rotSpeed, acceleration, deceleration;
	private int fadeSpeed, topSpeed;
	private boolean speedUp, slowDown, rotating;
	private int age = 0;
	private int whitevalue = 0;
	private int duration;	// counting frames before fading out
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int barWidth;

	public AdditivePropellerThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile,
			Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		red = 255;
		green = 255;
		blue = 255;
		rotating = true;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 20;
		this.physicalProps = physicalProps;
		duration = (lifespan*fps) - 150;
		barWidth = raster.width/6;
	}
	
	public AdditivePropellerThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile,
			Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		red = 255;
		green = 255;
		blue = 255;
		rotating = true;
		speedUp = true;
		slowDown = false;
		this.soundFile = soundFile;
		startSound = true;
		topSpeed = 20;
		this.physicalProps = physicalProps;
		duration = (lifespan*fps) - 150;
		barWidth = raster.width/6;
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
		raster.beginDraw();
		raster.noStroke();
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,raster.width,raster.height);
		raster.pushMatrix();
		raster.translate(raster.width/2,raster.height/2);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.fill(red,0,0);
		raster.rect(0,-barWidth,raster.width/2,barWidth);
		raster.rotate((float)(120 * Math.PI/180));
		raster.fill(0,green,0);
		raster.rect(0,-barWidth,raster.width/2,barWidth);
		raster.rotate((float)(120 * Math.PI/180));
		raster.fill(0,0,blue);
		raster.rect(0,-barWidth,raster.width/2,barWidth);
		raster.popMatrix();
		//raster.fill(255,255,255,whitevalue);
		//raster.rect(0,0,raster.width,raster.height);
		raster.endDraw();
		
		if(age > duration){
			speedUp = false;
			slowDown = true;
		}
		
		if(rotating){
			rotation += rotSpeed;
			if(speedUp){
				if(rotSpeed < topSpeed){
					rotSpeed += acceleration;
				}
				if(age > 90){
					if(whitevalue < 100){
						whitevalue += fadeSpeed;
					}
				}
			} else if(slowDown){
				rotSpeed -= deceleration;
				if(rotSpeed < 0){
					rotSpeed = 0;
				}
				if(whitevalue > 0){
					whitevalue -= fadeSpeed;
				}
				if(rotSpeed < 1){
					if(red > 1 || green > 1 || blue > 1){
						red = red - (int)(fadeSpeed*2.55);
						green = green - (int)(fadeSpeed*2.55);
						blue = blue - (int)(fadeSpeed*2.55);
					} else {
						cleanStop();
					}
				}
			}
		}
		
		age++;
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			//this.cleanStop();
			// slow down when sensor triggered off
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
			red = 255;
			green = 255;
			blue = 255;
		}
	}

}
