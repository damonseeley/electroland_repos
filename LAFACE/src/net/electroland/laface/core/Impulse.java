package net.electroland.laface.core;

import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.animation.Animation;

public class Impulse extends Thread{
	
	private long duration, startTime, dampDuration, dampStartTime;
	private int x, starty, targety;
	private double dampingTarget;
	private boolean impulseMode = true;
	private Wave wave;
	
	public Impulse(LAFACEMain main, int waveID, long duration, boolean leftSide){
		this.duration = duration;
		Animation a = main.getCurrentAnimation();
		if(a instanceof WaveShow){
			if(waveID >= 0){
				wave = ((WaveShow)a).getWave(waveID);
				//wave.setDamping(0);
				dampingTarget = 0.1;
				starty = 100;
				targety = 0;	// amplitude based on speed
				if(leftSide){
					x = 20;
				} else {
					x = 1040;
				}
			}
		}
		startTime = System.currentTimeMillis();
	}
	
	public void run(){
		// this will loop continually while sending impact events to the wave
		while(impulseMode){
			int y = (int)(((System.currentTimeMillis() - startTime)/(float)duration) * (targety-starty)) + starty;
			//wave.createImpact(x, y);
			wave.autoImpact(x, 0-y);
			if(System.currentTimeMillis() - startTime > duration){
				impulseMode = false;
				dampDuration = 3000;
			}
			
			try {
				sleep(5);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
			
		}
		
		/*
		try {
			sleep(dampDuration);
			//dampStartTime = System.currentTimeMillis();
			wave.setDamping(dampingTarget);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		*/
		
		// after sending a bunch of impact events, it will gradually adjust the damping
		//while(System.currentTimeMillis() - dampStartTime < dampDuration){
			//wave.setDamping((((System.currentTimeMillis() - startTime)/(float)duration) * dampingTarget));	// what happens during two impulses?
		//}
	}
	
}
