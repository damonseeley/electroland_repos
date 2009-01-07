package net.electroland.lafm.shows;

import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class AdditivePropellerThread extends ShowThread implements SensorListener{
	
	private int red, green, blue;
	private float rotation, rotSpeed, acceleration, deceleration;
	private int fadeSpeed;
	private boolean speedUp, slowDown, rotating;
	private PGraphics redRaster, greenRaster, blueRaster;
	private int length = 1;
	private int age = 0;
	private int whitevalue = 0;
	//private String soundFile;

	public AdditivePropellerThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, PGraphics redRaster,
			PGraphics greenRaster, PGraphics blueRaster, String soundFile) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.redRaster = redRaster;
		this.greenRaster = greenRaster;
		this.blueRaster = blueRaster;
		//this.soundFile = soundFile;
		red = 255;
		green = 255;
		blue = 255;
		rotating = true;
		speedUp = true;
		slowDown = false;
		if(soundManager != null){
			soundManager.playSimpleSound(soundFile, flower.getSoundChannel(), 1.0f, ID);
		}
	}
	
	public AdditivePropellerThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, float rotationSpeed, int fadeSpeed,
			float acceleration, float deceleration, PGraphics redRaster,
			PGraphics greenRaster, PGraphics blueRaster, String soundFile) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.rotation = 0;
		this.rotSpeed = rotationSpeed;
		this.fadeSpeed = fadeSpeed;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		this.redRaster = redRaster;
		this.greenRaster = greenRaster;
		this.blueRaster = blueRaster;
		//this.soundFile = soundFile;
		red = 255;
		green = 255;
		blue = 255;
		rotating = true;
		speedUp = true;
		slowDown = false;
		
		boolean[] channelsInUse = new boolean[6];		// null array of sound channels
		for(int n=0; n<channelsInUse.length; n++){
			channelsInUse[n] = false;
		}
		if(soundManager != null){
			Iterator <DMXLightingFixture> i = flowers.iterator();
			while (i.hasNext()){
				DMXLightingFixture flower = i.next();
				channelsInUse[flower.getSoundChannel()-1] = true;
			}
			for(int n=0; n<channelsInUse.length; n++){
				if(channelsInUse[n] != false){
					soundManager.playSimpleSound(soundFile, n+1, 1.0f, ID);
				}
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
		
		/*
		// doesn't rotate until propellers are fully extended
		if(length < 128){
			length += 3;
		} else if(length >= 128 && !rotating){
			length = 128;
			rotating = true;
		}
		*/
		
		if(length < 128){
			length += 5;
		} else if(length >= 128){
			length = 128;
		}
		
		redRaster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		redRaster.beginDraw();
		redRaster.noStroke();
		redRaster.fill(0,0,0,fadeSpeed);
		redRaster.rect(0,0,256,256);
		redRaster.translate(128,128);
		redRaster.rotate((float)(rotation * Math.PI/180));
		redRaster.fill(red,0,0);
		redRaster.rect(0,-10,length,20);
		redRaster.endDraw();
		
		greenRaster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		greenRaster.beginDraw();
		greenRaster.noStroke();
		greenRaster.fill(0,0,0,fadeSpeed);
		greenRaster.rect(0,0,256,256);
		greenRaster.translate(128,128);
		greenRaster.rotate((float)((rotation+120) * Math.PI/180));
		greenRaster.fill(0,green,0);
		greenRaster.rect(0,-10,length,20);
		greenRaster.endDraw();
		
		blueRaster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		blueRaster.beginDraw();
		blueRaster.noStroke();
		blueRaster.fill(0,0,0,fadeSpeed);
		blueRaster.rect(0,0,256,256);
		blueRaster.translate(128,128);
		blueRaster.rotate((float)((rotation+240) * Math.PI/180));
		blueRaster.fill(0,0,blue);
		blueRaster.rect(0,-10,length,20);
		blueRaster.endDraw();
		
		raster.colorMode(PConstants.RGB, 255, 255, 255, 100);
		raster.beginDraw();
		raster.noStroke();
		raster.background(0);	// wipes clean each
		raster.blend(redRaster, 0, 0, 256, 256, 0, 0, 256, 256, PConstants.ADD);
		raster.blend(greenRaster, 0, 0, 256, 256, 0, 0, 256, 256, PConstants.ADD);
		raster.blend(blueRaster, 0, 0, 256, 256, 0, 0, 256, 256, PConstants.ADD);
		raster.fill(255,255,255,whitevalue);
		raster.rect(0,0,256,256);
		raster.endDraw();
		
		if(rotating){
			rotation += rotSpeed;
			if(speedUp){
				rotSpeed += acceleration;
				if(age > 90){
					if(whitevalue < 100){
						whitevalue += 1;
					}
				}
			} else if(slowDown){
				rotSpeed -= deceleration;
				if(whitevalue > 0){
					whitevalue -= 1;
				}
				if(rotSpeed < 1){
					if(red > 1 || green > 1 || blue > 1){
						red = red - 15;
						green = green - 15;
						blue = blue - 15;
					} else {
						cleanStop();
					}
				}
			}
		}
		
		age++;
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			//this.cleanStop();
			// slow down when sensor triggered off
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
			red = 255;
			green = 255;
			blue = 255;
		}
	}

}
