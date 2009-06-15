package net.electroland.laface.core;

public class ImpulseThread extends Thread {
	
	private long duration, startTime;
	private int minDuration, maxDuration;
	private boolean running = false;
	public LAFACEMain main;
	
	public ImpulseThread(LAFACEMain main){
		this.main = main;
		minDuration = 3000;
		maxDuration = 8000;
		duration = (int)((maxDuration - minDuration)*Math.random() + minDuration);	
	}
	
	public void run(){
		running = true;
		startTime = System.currentTimeMillis();	
		// this will loop continually while sending impact events to the wave
		while(running){
			if(System.currentTimeMillis() - startTime > duration){
				if(Math.random() > 0.5){
					Impulse impulse = new Impulse(main, 0, 2000, true);
					impulse.start();
				} else {
					Impulse impulse = new Impulse(main, 0, 2000, false);
					impulse.start();
				}
				duration = (int)((maxDuration - minDuration)*Math.random() + minDuration);
				startTime = System.currentTimeMillis();	
			}
		}
	}

}
