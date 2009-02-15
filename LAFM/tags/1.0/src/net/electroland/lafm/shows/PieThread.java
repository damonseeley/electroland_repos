package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class PieThread extends ShowThread {
	
	private int red, green, blue;
	private int rotation;
	private float rotSpeed;
	private PImage texture;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private float gain;

	public PieThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, PImage texture, String soundFile,
			Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = 360 / (int)(lifespan*fps);
		this.texture = texture;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.gain = gain;
	}
	
	public PieThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, PImage texture, String soundFile,
			Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = 360 / (lifespan*fps);
		this.texture = texture;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
		this.gain = gain;
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
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(raster.width/2, raster.height/2);
		raster.fill(red,green,blue);
		raster.rectMode(PConstants.CENTER);
		raster.tint(red,green,blue);
		raster.rect(0,0,raster.width/8,raster.width/8);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.image(texture,0-texture.width,0-texture.height);
		raster.endDraw();
		
		if(rotation < 360){
			rotation += rotSpeed;
		} else {
			cleanStop();
		}
	}

}
