package net.electroland.laface.core;

import java.util.Vector;

import net.electroland.elvis.imaging.PresenceDetector;
import net.electroland.elvis.imaging.acquisition.axisCamera.AxisCamera;
import net.electroland.elvis.regions.PolyRegion;
import net.electroland.laface.core.LAFACEConfig;
import net.electroland.laface.util.SensorPair;

public class SensorThread extends Thread{
	boolean isRunning = true;
	AxisCamera cam;
	PresenceDetector detector;
	Vector<SensorPair>sensorPairs;
	//TrafficObserver trafficObserver;
	
	public SensorThread(PresenceDetector detector, AxisCamera cam, Vector<SensorPair>sensorPairs){
		this.detector = detector;
		this.cam = cam;
		this.sensorPairs = sensorPairs;
	}

	public void startNorthernCamSprite(long time, int lane){
		if (LAFACEConfig.TESTING){
			System.out.println("NORTH - NEW CAR in lane " + lane + " w/ time " + time);
		}
		
		// TODO start an Impulse within the current Wave sprite
		
		//register the new car with trafficobserver
		//trafficobserver.carDetected(true);
		
	}

	public void startSouthernSprite(long time, int lane){
		if (LAFACEConfig.TESTING){
			System.out.println("SOUTH - NEW CAR in lane " + lane + " w/ time " + time);
		}
		
		// TODO start an Impulse within the current Wave sprite

		//register the new car with trafficobserver
		//trafficobserver.carDetected(false);

	}
	
	public void run() {

		cam.start();
		detector.start(); // start detector before cam or else you'll get synchronization problems
		Vector<PolyRegion> regions = detector.getRegions();
		boolean[] triggers = new boolean[regions.size()];
		
		// init to false which should be done by java anyway
		for(int i =0; i < triggers.length; i++) {
			triggers[i] = false;
		}

		while(isRunning) {
			int i = 0;
			for(PolyRegion region : regions) { 	// check all the regions
				if(region.isTriggered) {  		// if triggered now
					if(! triggers[i]) { 		// but not previously
						
						triggers[i] = true;

						for (SensorPair pair : sensorPairs){
							if (pair.type == LAFACEConfig.SOUTH){
								//System.out.println("SOUTH " + region.name + " is triggered");										
							}
							if (pair.type == LAFACEConfig.NORTH){
								//System.out.println("NORTH " + region.name + " is triggered");										
							}
							if (region.id == pair.startSensorId){
								if (!pair.waiting){
									pair.startTime = System.currentTimeMillis();
									pair.waiting = true;
								}
							}else if (region.id == pair.endSensorId){									
								if (pair.waiting){
									long time = System.currentTimeMillis() - pair.startTime;
									if (time <= pair.threshold){
										
										time *= pair.tmultiplier;
										time = time < LAFACEConfig.UPPER_TIME_LIMIT ? time : LAFACEConfig.UPPER_TIME_LIMIT;
										time = time > LAFACEConfig.LOWER_TIME_LIMIT ? time : LAFACEConfig.LOWER_TIME_LIMIT;
										
										switch (pair.type){
										case(LAFACEConfig.NORTH):
											startNorthernCamSprite(time, pair.id);
											break;
										case(LAFACEConfig.SOUTH):
											startSouthernSprite(time, pair.id);
											break;
										}											
									}else{
										if (LAFACEConfig.TESTING){
											System.out.println("ignoring vehicle at speed " + time);
										}
									}
									pair.waiting = false;
								}
							}
						}
					}
				} else { 				// if not triggered now
					if(triggers[i]) { 	// but was previously
						//System.out.println(region.name + " is no longer triggered");
						triggers[i] = false;
					}					
				}
				i++;

			}
			// sleep a bit so we don't hammer the processor
			try { sleep(100); } catch (InterruptedException e) {};
		}
		cam.stopRunning();
		detector.stopRunning();
	}
}
