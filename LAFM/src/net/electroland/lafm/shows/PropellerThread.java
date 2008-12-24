package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class PropellerThread extends ShowThread {

	private int red, green, blue;					// normalized color value parameters
	private float rotation, rotSpeed;
	private int fadeSpeed;
	
	public PropellerThread(List<DMXLightingFixture> flowers, SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int showPriority, int red, int green, int blue, float rotationSpeed, int fadeSpeed) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
	}
	
	public PropellerThread(DMXLightingFixture flower, SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int showPriority, int red, int green, int blue, float rotationSpeed, int fadeSpeed) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
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
		raster.fill(red, green, blue);
		raster.rect(0,0,300,20);
		raster.endDraw();
		rotation += rotSpeed;
	}

}
