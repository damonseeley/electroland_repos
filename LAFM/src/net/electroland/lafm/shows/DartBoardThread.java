package net.electroland.lafm.shows;

import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;
import net.electroland.lafm.util.ColorScheme;

public class DartBoardThread extends ShowThread implements SensorListener{
	
	ColorScheme spectrum;
	float val1, val2, val3, offset;
	float speed, acceleration, deceleration;
	float[] color;
	int black = 0;
	boolean speedUp, slowDown;

	public DartBoardThread(DMXLightingFixture flower,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float offset, float acceleration, float deceleration) {
		super(flower, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.offset = offset;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		val1 = 0;
		val2 = val1 + offset;
		val3 = val2 + offset;
		speedUp = true;
		slowDown = false;
	}
	
	public DartBoardThread(List<DMXLightingFixture> flowers,
			SoundManager soundManager, int lifespan, int fps, PGraphics raster,
			String ID, int showPriority, ColorScheme spectrum, float speed,
			float offset, float acceleration, float deceleration) {
		super(flowers, soundManager, lifespan, fps, raster, ID, showPriority);
		this.spectrum = spectrum;
		this.speed = speed;
		this.offset = offset;
		this.acceleration = acceleration;
		this.deceleration = deceleration;
		val1 = 0;
		val2 = val1 + offset;
		val3 = val2 + offset;
		speedUp = true;
		slowDown = false;
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
		float[] colora = spectrum.getColor(val1);
		raster.fill(colora[0],colora[1],colora[2]);
		raster.ellipse(0,0,250,250);
		float[] colorb = spectrum.getColor(val2);
		raster.fill(colorb[0],colorb[1],colorb[2]);
		raster.ellipse(0,0,150,150);
		float[] colorc = spectrum.getColor(val3);
		raster.fill(colorc[0],colorc[1],colorc[2]);
		raster.ellipse(0,0,50,50);
		raster.fill(0,0,0,black);
		raster.rect(-128,-128,256,256);
		raster.endDraw();
		
		if(val1 >= 1){
			val1 = 0;
		} else {
			val1 += speed;
		}
		if(val2 >= 1){
			val2 = 0;
		} else {
			val2 += speed;
		}
		if(val3 >= 1){
			val3 = 0;
		} else {
			val3 += speed;
		}
		
		if(speedUp){
			speed += acceleration;
		} else if(slowDown){
			if(speed > 0){
				speed -= deceleration;
			}
			if(speed < 0.01){
				if(black >= 100){
					cleanStop();					
				} else {
					black += 5;
				}
			}
		}
	}
	
	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (this.getFlowers().contains(eventFixture) && !isOn){
			//this.cleanStop();
			// potentially slow down when sensor triggered off
			speedUp = false;
			slowDown = true;
		} else if(this.getFlowers().contains(eventFixture) && isOn){
			// reactivate
			speedUp = true;
			slowDown = false;
			black = 0;
		}
	}

}
