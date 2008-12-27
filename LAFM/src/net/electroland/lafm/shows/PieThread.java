package net.electroland.lafm.shows;

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

	public PieThread(DMXLightingFixture flower, SoundManager soundManager,
			int lifespan, int fps, PGraphics raster, String ID, int showPriority,
			int red, int green, int blue, PImage texture) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.red = red;
		this.green = green;
		this.blue = blue;
		this.rotation = 0;
		this.rotSpeed = 360 / (lifespan*fps);
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
		raster.beginDraw();
		raster.noStroke();
		raster.translate(128, 128);
		raster.rotate((float)(rotation * Math.PI/180));
		raster.tint(red,green,blue);
		raster.image(texture,0-texture.width,0-texture.height);
		raster.endDraw();
		
		if(rotation < 360){
			rotation += rotSpeed;
		} else {
			complete(raster);
		}
	}

}
