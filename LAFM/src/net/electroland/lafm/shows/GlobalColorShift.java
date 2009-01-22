package net.electroland.lafm.shows;

import java.util.List;
import java.util.Properties;

import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class GlobalColorShift extends ShowThread {
	
	private ColorScheme spectrum;
	private float speed, offset;
	private float[] color;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;

	public GlobalColorShift(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float offset, String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.offset = offset;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.startSound = true;
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
		
		color = spectrum.getColor(offset);
		raster.beginDraw();
		raster.background(color[0], color[1], color[2]);
		raster.endDraw();
		if(offset >= 1){
			offset = 0;
		} else {
			offset += speed;
		}
	}

}
