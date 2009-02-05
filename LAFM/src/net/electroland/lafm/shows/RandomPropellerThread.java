package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class RandomPropellerThread extends ShowThread implements SensorListener {
	
	private float rotSpeed, acceleration, deceleration;
	private int fadeSpeed, topSpeed;
	private boolean speedUp, slowDown;
	private boolean startSound;
	private String soundFile;
	private Properties physicalProps;
	private int propellerCount = 0;	// unique ID
	private int numPropellers = 0;		// number in play
	private int age = 0;
	private int alpha = 0;
	private boolean fadeOut = false;
	private int duration;	// counting frames before fading out
	private ColorScheme spectrum;
	private ConcurrentHashMap<Integer,Propeller> propellers;
	private float gain;

	public RandomPropellerThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile, Properties physicalProps, float gain) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.gain = gain;
		speedUp = true;
		slowDown = false;
		topSpeed = 15;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/(fadeSpeed/4));
		propellers = new ConcurrentHashMap<Integer,Propeller>();
		int lastRot = (int)(Math.random()*360);
		float lastColor = (float)Math.random();
		for(int i=0; i<2; i++){
			propellers.put(propellerCount, new Propeller(propellerCount, i*30, lastRot, lastColor));
			propellerCount++;
			numPropellers++;
			lastRot += 90;
			lastColor += 0.5;
			if(lastRot > 360){
				lastRot -= 360;
			}
			if(lastColor > 1){
				lastColor -= 1;
			}
		}
		startSound = true;
	}
	
	public RandomPropellerThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, String soundFile, Properties physicalProps, float gain) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);

		this.spectrum = spectrum;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.soundFile = soundFile;
		this.physicalProps = physicalProps;
		this.gain = gain;
		speedUp = true;
		slowDown = false;
		topSpeed = 15;
		duration = ((int)(lifespan/1000.0f)*fps) - (int)(100/(fadeSpeed/4));
		propellers = new ConcurrentHashMap<Integer,Propeller>();
		int lastRot = (int)(Math.random()*360);
		float lastColor = (float)Math.random();
		for(int i=0; i<2; i++){
			propellers.put(propellerCount, new Propeller(propellerCount, i*30, lastRot, lastColor));
			propellerCount++;
			numPropellers++;
			lastRot += 90;
			lastColor += 0.5;
			if(lastRot > 360){
				lastRot -= 360;
			}
			if(lastColor > 1){
				lastColor -= 1;
			}
		}
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
		if(startSound){
			super.playSound(soundFile, gain, physicalProps);
			startSound = false;
			raster.beginDraw();
			raster.background(255,255,255);	// flashes white at beginning
			raster.endDraw();
		}
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.rectMode(PConstants.CENTER);
		raster.beginDraw();
		raster.noStroke();
		raster.translate(raster.width/2, raster.height/2);
		raster.fill(0,0,0,fadeSpeed);
		raster.rect(0,0,raster.width,raster.height);
		Iterator<Propeller> i = propellers.values().iterator();
		while (i.hasNext()){
			Propeller b = i.next();
			b.draw(raster);
		}
		raster.endDraw();
		
		/*
		if(numPropellers < 2){			// add another propeller
			//if(Math.random() > 0.7){
				Iterator<Propeller> iter = propellers.values().iterator();
				if(iter.hasNext()){
					Propeller prop = iter.next();
					int newrot = prop.rotation + 90;
					if(newrot > 360){
						newrot -= 360;
					}
					float newcolor = prop.colorPoint + 0.25f + (float)(Math.random()*0.5f);
					if(newcolor > 1){
						newcolor -= 1;
					}
					propellers.put(propellerCount, new Propeller(propellerCount, 0, newrot, newcolor));
				} else {
					propellers.put(propellerCount, new Propeller(propellerCount, 0));	// hopefully this never runs
				}
				propellerCount++;
				numPropellers++;
			//}
		}
		*/
		
		if(age > duration){
			fadeOut = true;
		}
		if(fadeOut){
			if(alpha < 100){
				alpha += fadeSpeed/4;
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
			} 
			if(alpha >= 100){
				raster.fill(0,0,0,alpha);
				raster.rect(0,0,raster.width,raster.height);
				cleanStop();
			}
		}
		age++;
		
		if(speedUp){
			if(rotSpeed < topSpeed){
				rotSpeed += acceleration;
			}
		} else if(slowDown){
			if(rotSpeed <= 0){
				rotSpeed = 0;
			} else {
				rotSpeed -= deceleration;
			}
			if(rotSpeed < 1){
				fadeOut = true;
			}
		}

	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
			alpha = 100;
		}
	}
	
	
	
	
	private class Propeller{
		
		private int alpha, startDelay, delayCounter;
		//private int id, age, oldAge;
		//private float speed;
		private float[] color;
		public float colorPoint;
		public int rotation;
		
		private Propeller(int id, int startDelay){
			//this.id = id;
			this.startDelay = startDelay;
			//this.speed = rotSpeed;	// starting speed
			rotation = (int)(Math.random()*360);
			//age = 0;
			alpha = 100;
			//oldAge = (int)(Math.random()*450)+150;	// when to fade out (5-20 seconds)
			colorPoint = (float)Math.random();
			color = spectrum.getColor(colorPoint);
			delayCounter = 0;
		}
		
		private Propeller(int id, int startDelay, int rotation, float colorpoint){
			//this.id = id;
			this.startDelay = startDelay;
			//this.speed = rotSpeed;	// starting speed
			this.rotation = rotation;
			this.colorPoint = colorpoint;
			//age = 0;
			alpha = 100;
			//oldAge = (int)(Math.random()*450)+150;	// when to fade out (5-20 seconds)
			color = spectrum.getColor(colorPoint);
			delayCounter = 0;
		}
		
		public void draw(PGraphics raster){
			if(delayCounter < startDelay){
				delayCounter++;
			} else {
				/*
				if(age > oldAge){
					if(alpha > 0){
						alpha -= fadeSpeed;
					} else {
						numPropellers--;
						propellers.remove(id);
					}
				}
				age++;
				*/
				colorPoint += 0.01;
				if(colorPoint > 1){
					colorPoint -= 1;
				}
				color = spectrum.getColor(colorPoint);
				rotation += rotSpeed;
				
				/*
				if(speedUp){
					if(speed < topSpeed){
						speed += acceleration;
					}
				} else if(slowDown){
					if(speed <= 0){
						speed = 0;
					} else if(speed < 1){
						fadeOut = true;
					} else {
						speed -= deceleration;
					}
				}
				*/
				
				raster.pushMatrix();
				raster.rotate((float)(rotation * Math.PI/180));
				raster.fill(color[0], color[1], color[2], alpha);
				raster.rect(0,0,raster.width + raster.width/5,raster.height/7);
				raster.popMatrix();
			}
		}
	}

}
