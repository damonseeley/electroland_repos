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

public class SpiralThread extends ShowThread implements SensorListener{
	
	private int red, green, blue;
	private float rotation, rotSpeed;
	private int fadeSpeed, spriteWidth;
	private float spiralTightness, currentTightness;
	private PImage texture;
	private boolean startSound, fadeOut;
	private String soundFile;
	private Properties physicalProps;
	int fadeOutSpeed = 3;
	int alpha = 0;
	int age = 0;
	private boolean loop = true;
	private int duration;	// counting frames before fading out
	private float gain;

	public SpiralThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue,  float rotationSpeed, int fadeSpeed,
			float spiralTightness, int spriteWidth, PImage texture, String soundFile,
			Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.spiralTightness = spiralTightness;
		this.currentTightness = 0;
		this.spriteWidth = spriteWidth;
		this.texture = texture;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.gain = gain;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
	}
	
	public SpiralThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue,  float rotationSpeed, int fadeSpeed,
			float spiralTightness, int spriteWidth, PImage texture, String soundFile,
			Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.spiralTightness = spiralTightness;
		this.currentTightness = 0;
		this.spriteWidth = spriteWidth;
		this.texture = texture;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.gain = gain;
		fadeOut = false;
		duration = (lifespan*fps) - (100/fadeSpeed);
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
		raster.tint(red,green,blue);
		raster.image(texture,0-(spriteWidth/2),currentTightness-(spriteWidth/2),spriteWidth,spriteWidth);
		
		if(age > duration){
			loop = false;
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeOutSpeed;
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
			} else {
				cleanStop();
			}
		}
		age++;
		
		raster.endDraw();
		rotation += rotSpeed;
		currentTightness += spiralTightness;
		if(currentTightness >= (raster.width/2)){
			if(loop){
				spiralTightness = 0 - spiralTightness;
			} else {
				fadeOut = true;	// should turn true before here, but just in case
			}
		} else if(currentTightness <= 0){
			spiralTightness = 0 - spiralTightness;
			currentTightness = spiralTightness;
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			loop = false;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			loop = true;
			fadeOut = false;
			alpha = 0;
		}
	}

}
