package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class FireworksThread extends ShowThread {
	
	ColorScheme spectrum;
	float speed, frequency;
	PImage texture;
	ConcurrentHashMap<Integer,Firework> fireworks;
	int fireworkCount = 0;
	float fadeSpeed = 10;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;

	public FireworksThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float frequency, PImage texture, String soundFile, Properties physicalProps) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.frequency = frequency;		// 0-1, odds of creating a new firework
		this.texture = texture;
		fireworks = new ConcurrentHashMap<Integer,Firework>();
		fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
		fireworkCount++;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
	}
	
	public FireworksThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float frequency, PImage texture, String soundFile, Properties physicalProps) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.frequency = frequency;
		this.texture = texture;
		fireworks = new ConcurrentHashMap<Integer,Firework>();
		fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
		fireworkCount++;
		this.soundFile = soundFile;
		startSound = true;
		this.physicalProps = physicalProps;
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
		
		if(Math.random() > frequency){
			fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
			fireworkCount++;
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.background(0);
		Iterator<Firework> i = fireworks.values().iterator();
		while (i.hasNext()){
			Firework f = i.next();
			f.draw(raster);
		}
		raster.endDraw();
	}
	
	
	
	
	
	public class Firework{
		
		int x, y, id, age;
		float diameter;
		float[] color;
		float alpha;
		
		public Firework(int id, float[] color, int width, int height){
			this.id = id;
			this.color = color;
			alpha = 100;
			age = 0;
			x = (int)(Math.random()*width);
			y = (int)(Math.random()*height);
			diameter = 5;
		}
		
		public void draw(PGraphics raster){
			raster.tint(color[0], color[1], color[2], alpha);
			raster.image(texture, x - diameter/2, y - diameter/2, diameter, diameter);
			diameter += speed;
			if(age > 15){
				if(alpha < 1){
					fireworks.remove(id);
				} else {
					alpha -= fadeSpeed;
				}
			} else {
				age++;
			}
		}
	}

}
