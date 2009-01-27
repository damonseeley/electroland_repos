package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class LightGroupTestThread extends ShowThread {
	
	PImage texture;

	public LightGroupTestThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, PImage texture) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.texture = texture;
	}
	
	public LightGroupTestThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, PImage texture) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.texture = texture;
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
		raster.image(texture,0,0,raster.width,raster.height);
		raster.endDraw();
	}

}
