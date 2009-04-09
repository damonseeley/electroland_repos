package net.electroland.laface.core;

import net.electroland.laface.util.SimpleTimer;

/**
 * 
 * @author Damon
 * This object receives car detections from SensorThreads determines whether the app
 * should be in text or sprite mode accordingly.  The mode is not explicitly set by
 * this object, rather read passively from within animationmanager.
 *
 * Appropriated from NoHo and modified by Aaron Siegel.
 *
 */


public class TrafficObserver {
	
	public boolean useTextMode; // whether the app should be in text or sprite mode

	// counter for sprite mode triggering
	protected int consecutiveCarCount;
	protected SimpleTimer CarTriggerTimeout;

	//car counters, for my interest
	protected int northCars = 0;
	protected int southCars = 0;

	
	public TrafficObserver() {
		consecutiveCarCount = 0;
		CarTriggerTimeout = new SimpleTimer(LAFACEConfig.CARSTRIGGERTIMEOUT);
	}
	
	public void carDetected(boolean northCar){
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
	
}
