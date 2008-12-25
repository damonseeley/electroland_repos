package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class SpiralThread extends ShowThread {
	
	private int red, green, blue;
	private float rotation, rotSpeed;
	private int fadeSpeed, spriteWidth;
	private float spiralTightness, currentTightness;
	private PImage texture;

	public SpiralThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue,  float rotationSpeed, int fadeSpeed,
			float spiralTightness, int spriteWidth, PImage texture) {
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
	}
	
	public SpiralThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority) {
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
		raster.rotate((float)(rotation * Math.PI/180));
		//raster.fill(red, green, blue);
		//raster.rect(0,currentTightness,spriteWidth,spriteWidth);
		raster.tint(red,green,blue);
		raster.image(texture,0-(spriteWidth/2),currentTightness-(spriteWidth/2),spriteWidth,spriteWidth);
		raster.endDraw();
		rotation += rotSpeed;
		currentTightness += spiralTightness;
		if(currentTightness >= (raster.width/2) + spriteWidth/2){
			currentTightness = 0;
		}
	}

}
