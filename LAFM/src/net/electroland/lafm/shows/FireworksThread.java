package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class FireworksThread extends ShowThread implements SensorListener{
	
	private ColorScheme spectrum;
	private float speed, frequency;
	private PImage texture;
	private ConcurrentHashMap<Integer,Firework> fireworks;
	private int fireworkCount = 0;
	private float fadeSpeed = 10;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int totalFrames;
	private int age = 0;
	private int startDelay, delayCount;
	private int alpha = 0;
	private boolean fadeOut;
	private boolean interactive;
	private int fadeOutSpeed = 3;
	private int duration;	// counting frames before fading out
	private float frequencySlowRate;
	private int backgroundColor = 255;

	public FireworksThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float frequency, PImage texture, String soundFile, Properties physicalProps,
			int startDelay, boolean interactive) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.frequency = frequency;		// 0-1, odds of creating a new firework
		this.texture = texture;
		fireworks = new ConcurrentHashMap<Integer,Firework>();
		fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
		fireworkCount++;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.totalFrames = lifespan*fps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		this.interactive = interactive;
		delayCount = 0;
		duration = (lifespan*fps) - (100/fadeOutSpeed);
		frequencySlowRate = (0.9f - frequency)/duration;	// add this to frequency
		startSound = true;
	}
	
	public FireworksThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float frequency, PImage texture, String soundFile, Properties physicalProps,
			int startDelay, boolean interactive) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.frequency = frequency;
		this.texture = texture;
		fireworks = new ConcurrentHashMap<Integer,Firework>();
		fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
		fireworkCount++;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.totalFrames = lifespan*fps;
		this.startDelay = (int)((startDelay/1000.0f)*fps);
		this.interactive = interactive;
		delayCount = 0;
		duration = (lifespan*fps) - (100/fadeOutSpeed);
		frequencySlowRate = (0.9f - frequency)/duration;	// add this to frequency
		startSound = true;
	}

	@Override
	public void complete(PGraphics raster) {
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	@Override
	public void doWork(PGraphics raster) {
		if(delayCount >= startDelay){
			if(startSound){
				super.playSound(soundFile, physicalProps);
				startSound = false;
			}
			
			if(age < totalFrames - 30){			// stop introducing fireworks at end
				if(Math.random() > frequency){
					fireworks.put(fireworkCount, new Firework(fireworkCount, spectrum.getColor((float)Math.random()), raster.width, raster.height));
					fireworkCount++;
				}
			}
			
			raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
			//raster.rectMode(PConstants.CENTER);
			raster.beginDraw();
			raster.noStroke();
			raster.background(backgroundColor);
			if(backgroundColor > 0){
				backgroundColor -= 10;
			}
			if(backgroundColor < 0){
				backgroundColor = 0;
			}
			Iterator<Firework> i = fireworks.values().iterator();
			while (i.hasNext()){
				Firework f = i.next();
				f.draw(this, raster);
			}
			if(age > duration){
				fadeOut = true;
			}
			if(fadeOut && age > 30){	// minimum duration to stay on
				if(alpha < 100){
					alpha += fadeOutSpeed;
					raster.fill(0,0,0,alpha);
					raster.rect(0,0,raster.width,raster.height);
				} 
				if(alpha >= 100){
					raster.fill(0,0,0,alpha);
					raster.rect(0,0,raster.width,raster.height);
					cleanStop();
				}
			}
			raster.endDraw();
			frequency += frequencySlowRate;
			//age++;
		} else {
			delayCount++;
			//super.resetLifespan();
		}
		age++;
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if(interactive){
			if (this.getFlowers().contains(eventFixture) && !isOn){
				// fade out
				fadeOut = true;
			} else if(this.getFlowers().contains(eventFixture) && isOn){
				// reactivate
				fadeOut = false;
				alpha = 0;
				//age = 0;
			}
		}
	}
	
	
	
	
	
	public class Firework{
		
		int x, y, id, age;
		float diameter;
		float[] color;
		float alpha;
		boolean startExplosionSound;
		
		public Firework(int id, float[] color, int width, int height){
			this.id = id;
			this.color = color;
			alpha = 100;
			age = 0;
			x = (int)(Math.random()*width);
			y = (int)(Math.random()*height);
			diameter = 5;
			startExplosionSound = true;
		}
		
		public void draw(FireworksThread parent, PGraphics raster){
			if(startExplosionSound && interactive){
				parent.playSound(soundFile, physicalProps);
				startExplosionSound = false;
			}
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
