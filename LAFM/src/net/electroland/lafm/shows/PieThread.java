package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;

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
	private int cycles;

	public PieThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, PImage texture, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = 360 / (int)(lifespan*fps);
		this.texture = texture;
		cycles = 0;
		if(soundManager != null){
			soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
		}
	}
	
	public PieThread(List<DMXLightingFixture> flowers, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, PImage texture, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = 360 / (lifespan*fps);
		this.texture = texture;
		cycles = 0;
		if(soundManager != null){
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				DMXLightingFixture flower = i.next();
				soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
			}
		}
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
		raster.beginDraw();
		raster.noStroke();
		raster.translate(128, 128);
		raster.fill(red,green,blue);
		raster.rectMode(PConstants.CENTER);
		raster.tint(red,green,blue);
		raster.rect(0,0,30,30);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.image(texture,0-texture.width,0-texture.height);
		raster.endDraw();
		
		if(rotation < 360){
			rotation += rotSpeed;
		} else {
			/*
			if(cycles < 3){
				cycles++;
				rotation = 0;
				red = (int)(Math.random()*255);		// random color may suck
				green = (int)(Math.random()*255);
				blue = (int)(Math.random()*255);
			} else {
				cleanStop();
			}
			*/
			cleanStop();
		}
	}

}
