package net.electroland.noho.core;

import net.electroland.noho.util.SimpleTimer;

/**
 * 
 * @author Damon
 * This object receives car detections from SensorThreads determines whether the app
 * should be in text or sprite mode accordingly.  The mode is not explicitly set by
 * this object, rather read passively from within animationmanager.
 *
 */


public class TrafficObserver {
	
	public boolean useTextMode; // whether the app should be in text or sprite mode
	
	protected SimpleTimer SpritesMaxTime;
	protected SimpleTimer SpritesMinTime;
	protected SimpleTimer SpritesTimeout;

	// counter for sprite mode triggering
	protected int consecutiveCarCount;
	protected SimpleTimer CarTriggerTimeout;
	
	
	//car counters, for my interest
	protected int northCars = 0;
	protected int southCars = 0;

	
	public TrafficObserver() {
		useTextMode = true;
		SpritesMaxTime = new SimpleTimer(NoHoConfig.SPRITESMAXTIME);
		SpritesMinTime = new SimpleTimer(NoHoConfig.SPRITESMINTIME);
		SpritesTimeout = new SimpleTimer(NoHoConfig.SPRITESTIMEOUT);
		
		consecutiveCarCount = 0;
		CarTriggerTimeout = new SimpleTimer(NoHoConfig.CARSTRIGGERTIMEOUT);
	}
	
	
	
	public void process() {
		//
		if (useTextMode){
			// checking for traffic
			if (consecutiveCarCount >= NoHoConfig.CARSTRIGGERTHRESH){
				switchToSpriteMode();
			}
		} else {
			// make sure sprites have run for a min time
			if (SpritesMinTime.isDone()){
				// if we are outside of sprites max time just switch no matter what
				if (SpritesMaxTime.isDone()){
					System.out.println("Sprite mode exceeds MAX time");
					switchToTextMode();
				} else {
					// check to see if sprite mode has timed out due to lack of cars
					if (SpritesTimeout.isDone()){
						System.out.println("Sprite mode TIMED OUT");
						switchToTextMode();
					}
				}
				
			}
			// checking for a lack of traffic
		}

	}
	
	private void switchToTextMode() {
		System.out.println("Switching to TEXT mode");
		useTextMode = true;
		consecutiveCarCount = 0;
		CarTriggerTimeout.resetTimer();
	}
	
	private void switchToSpriteMode() {
		System.out.println("Switching to SPRITE mode");
		useTextMode = false;
		SpritesMaxTime.resetTimer();
		SpritesMinTime.resetTimer();
		SpritesTimeout.resetTimer();
		
	}
	
	public void resetAll() {
		useTextMode = true;
		SpritesMaxTime.resetTimer();
		SpritesMinTime.resetTimer();
		SpritesTimeout.resetTimer();
		consecutiveCarCount = 0;
		CarTriggerTimeout.resetTimer();
	}
	
	public void carDetected(boolean northCar){
		if (useTextMode){
			//System.out.println("car detected");
			if (CarTriggerTimeout.isDone()) {
				// if the trigger timeout is done the counter should be reset to 
				// 1 and the timeout timer is reset.
				consecutiveCarCount = 1;
				CarTriggerTimeout.resetTimer();
			} else {
				// if the trigger timeout counter is NOT done then we
				// inc the car count and continue
				consecutiveCarCount++;
				System.out.println(consecutiveCarCount + " cars in this detection period");
			}
		} else {
			// for right now the sprite timeout resets if ANY car passes
			// this probably means too much sprite mode
			SpritesTimeout.resetTimer();
		}


		
		// car counting by direction
		if (northCar){
			northCars++;
		} else {
			southCars++;
		}
		
		//every once in a while print out the car counts
		if (northCars > 0 && northCars % 100 == 0){
			System.out.println(northCars + " southbound cars have passed during this session");
		}
		if (southCars > 0 && southCars % 100 == 0){
			System.out.println(southCars + " northbound cars have passed during this session");
		}
		
	}
	
	// the primary interface for other objects
	public boolean useTextMode() {
		return useTextMode;
	}
	
	
	
	

}
