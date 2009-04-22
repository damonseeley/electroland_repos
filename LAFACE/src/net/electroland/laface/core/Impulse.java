package net.electroland.laface.core;

import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.animation.Animation;

public class Impulse extends Thread{
	
	private long duration, startTime;
	private int x, starty, targety;
	private boolean impulseMode = true;
	private Wave wave;
	
	public Impulse(LAFACEMain main, int waveID, long duration, boolean leftSide){
		this.duration = duration;
		Animation a = main.getCurrentAnimation();
		if(a instanceof WaveShow){
			if(waveID >= 0){
				wave = ((WaveShow)a).getWave(waveID);
				//wave.setDamping(0);
				//dampingTarget = 0.1;
				starty = 80;
				targety = 0;	// amplitude based on speed
				if(leftSide){
					x = 0;
				} else {
					x = 1048;
				}
			}
		}
		startTime = System.currentTimeMillis();
	}
	
	public Impulse(LAFACEMain main, int waveID, long duration, boolean leftSide, int starty, int targety){
		this.duration = duration;
		Animation a = main.getCurrentAnimation();
		if(a instanceof WaveShow){
			if(waveID >= 0){
				wave = ((WaveShow)a).getWave(waveID);
				//wave.setDamping(0);
				//dampingTarget = 0.1;
				this.starty = starty;
				this.targety = targety;	// amplitude based on speed
				if(leftSide){
					x = 0;
				} else {
					x = 1048;
				}
			}
		}
		startTime = System.currentTimeMillis();
	}
	
	public void run(){
		// this will loop continually while sending impact events to the wave
		while(impulseMode){
			int y = (int)(((System.currentTimeMillis() - startTime)/(float)duration) * (targety-starty)) + starty;
			wave.createImpact(x, y);
			//wave.autoImpact(x, 0-y);
			if(System.currentTimeMillis() - startTime > duration){
				impulseMode = false;
			}
		}
	}
	
}
