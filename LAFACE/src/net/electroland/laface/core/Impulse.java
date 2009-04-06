package net.electroland.laface.core;

import net.electroland.laface.shows.WaveShow;
import net.electroland.laface.sprites.Wave;
import net.electroland.lighting.detector.animation.Animation;

public class Impulse extends Thread{
	
	private long duration, startTime, dampDuration, dampStartTime;
	private int startx, starty, targetx, targety;
	private double dampingTarget;
	private boolean impulseMode = true;
	private Wave wave;
	
	public Impulse(LAFACEMain main, int waveID, long duration, boolean leftSide){
		this.duration = duration;
		Animation a = main.getCurrentAnimation();
		if(a instanceof WaveShow){
			if(waveID >= 0){
				wave = ((WaveShow)a).getWave(waveID);
				wave.setDamping(0);
				dampingTarget = 0.1;
				if(leftSide){		// TODO this is totally kludge
					startx = 0;
					starty = 133;
					targetx = 133;
					targety = -100;	// amplitude based on speed
				} else {
					startx = 1048;
					starty = 133;
					targetx = 915;
					targety = -100;	// amplitude based on speed
				}
			}
		}
		startTime = System.currentTimeMillis();
	}
	
	public void run(){
		// this will loop continually while sending impact events to the wave
		while(impulseMode){
			int x = (int)(((System.currentTimeMillis() - startTime)/(float)duration) * (targetx-startx)) + startx;
			int y = (int)(((System.currentTimeMillis() - startTime)/(float)duration) * (targety-starty)) + starty;
			//System.out.println((System.currentTimeMillis() - startTime)/(float)duration);
			wave.createImpact(x, y);
			if(System.currentTimeMillis() - startTime > duration){
				impulseMode = false;
				dampDuration = 3000;
			}
			try {
				sleep(33);
			} catch (InterruptedException e) {
				// TODO Auto-generated catch block
				e.printStackTrace();
			}
		}
		
		try {
			sleep(dampDuration);
			//dampStartTime = System.currentTimeMillis();
			wave.setDamping(dampingTarget);
		} catch (InterruptedException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		// after sending a bunch of impact events, it will gradually adjust the damping
		//while(System.currentTimeMillis() - dampStartTime < dampDuration){
			//wave.setDamping((((System.currentTimeMillis() - startTime)/(float)duration) * dampingTarget));	// what happens during two impulses?
		//}
	}
	
}
