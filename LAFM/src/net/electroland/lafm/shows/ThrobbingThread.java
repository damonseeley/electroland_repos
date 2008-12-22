package net.electroland.lafm.shows;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.detector.DMXLightingFixture;
import net.electroland.lafm.core.SensorListener;
import net.electroland.lafm.core.ShowThread;
import net.electroland.lafm.core.SoundManager;

public class ThrobbingThread extends ShowThread implements SensorListener{

	private float red, green, blue;					// normalized color value parameters
	private float brightness;							// intensity changed in throbbing
	private int holdon, holdoff;						// timing parameters in milliseconds
	private float fadeinspeed, fadeoutspeed;			// brightness change increments
	private long lastChange;
	private int state;									// 0 = fade in, 1 = hold on, 2 = fade out, 3 = hold off
	public String name = "ThrobbingThread";

	public ThrobbingThread(DMXLightingFixture flower, SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int red, int blue, int green, int fadein, int fadeout, int holdon, int holdoff) {
		super(flower, soundManager, lifespan, fps, raster, ID);
		this.red = (float)(red/255.0);
		this.green = (float)(green/255.0);
		this.blue = (float)(blue/255.0);
		this.fadeinspeed = 255 / ((float)(fadein/1000.0)*fps);
		this.fadeoutspeed = 255 / ((float)(fadeout/1000.0)*fps);
		System.out.println(fadeinspeed +" "+ fadeoutspeed);
		this.holdon = holdon;
		this.holdoff = holdoff;
		this.brightness = 0;
		this.state = 0;
		raster.colorMode(PConstants.RGB, 255, 255, 255);
	}
	
	public ThrobbingThread(DMXLightingFixture[] flowers, SoundManager soundManager, int lifespan, int fps, PGraphics raster, String ID, int red, int blue, int green, int fadein, int fadeout, int holdon, int holdoff) {
		super(flowers, soundManager, lifespan, fps, raster, ID);
		this.red = (float)(red/255.0);
		this.green = (float)(green/255.0);
		this.blue = (float)(blue/255.0);
		this.fadeinspeed = (int)((fadein/1000.0)*fps);
		this.fadeoutspeed = (int)((fadeout/1000.0)*fps);
		this.holdon = holdon;
		this.holdoff = holdoff;
		this.brightness = 0;
		this.state = 0;
		raster.colorMode(PConstants.RGB, 255, 255, 255);
	}

	@Override
	public void complete(PGraphics raster) {
		raster.background(0);

	}

	@Override
	public void doWork(PGraphics raster) {
		//System.out.println(this.brightness);
		
		if(state == 0 && brightness < 255){				// fade in
			brightness += fadeinspeed;
		} else if(state == 0 && brightness >= 255){	// switch to holding on
			state = 1;
			lastChange = System.currentTimeMillis();
		} else if(state == 1){							// hold on
			if(System.currentTimeMillis() - lastChange >= holdon){
				state = 2;
			}
		} else if(state == 2 && brightness > 0){		// fade out
			brightness -= fadeoutspeed;
		} else if(state == 2 && brightness <= 0){		// switch to holding off
			state = 3;
			lastChange = System.currentTimeMillis();
		} else if(state == 3){							// hold off
			if(System.currentTimeMillis() - lastChange >= holdoff){
				state = 0;
			}
		}
		
		raster.beginDraw();
		raster.background(red*brightness, green*brightness, blue*brightness);
		raster.endDraw();
	}

	public void sensorEvent(DMXLightingFixture eventFixture, boolean isOn) {
		// assumes that this thread is only used in a single thread per fixture
		// environment (thus this.getFlowers() is an array of 1)
		if (eventFixture == this.getFlowers()[0] && !isOn){
			this.cleanStop();
		}
	}
}
