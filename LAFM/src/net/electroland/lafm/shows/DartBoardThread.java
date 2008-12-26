package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class DartBoardThread extends ShowThread {
	
	ColorScheme spectrum;
	float val1, val2, val3;

	public DartBoardThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		val1 = 0.1f;
		val2 = 0.4f;
		val3 = 0.8f;
	}
	
	public DartBoardThread(List<DMXLightingFixture> flowers,
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
		raster.ellipseMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(128, 128);
		float[] color = spectrum.getColor(val1);
		raster.fill(color[0],color[1],color[2]);
		raster.ellipse(0,0,250,250);
		color = spectrum.getColor(val2);
		raster.fill(color[0],color[1],color[2]);
		raster.ellipse(0,0,150,150);
		color = spectrum.getColor(val3);
		raster.fill(color[0],color[1],color[2]);
		raster.ellipse(0,0,50,50);
		raster.endDraw();
		if(val1 > 1){
			val1 = 0;
		} else {
			val1 += 0.01;
		}
		if(val2 > 1){
			val2 = 0;
		} else {
			val2 += 0.01;
		}
		if(val3 > 1){
			val3 = 0;
		} else {
			val3 += 0.01;
		}
	}

}
