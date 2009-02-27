package net.electroland.enteractive.core;

import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;

/**
 * Initiates the program and controls show changes
 * @author asiegel
 */

public class Main implements TimedEventListener, SensorListener {
	// TODO: This class must implement ShowThreadListener

	SoundManager soundManager;
	
	public Main(){
		loadProperties();
		soundManager = new SoundManager(2,2,null);
		// TODO: buffer sound files from properties
		// TODO: Start sensorManager
		// TODO: Start syncThread
	}
	
	private void loadProperties(){
		// TODO: Load properties from file
	}

	@Override
	public void timedEvent(TimedEvent event) {
		// TODO: Trigger scheduled show changes here
	}

	@Override
	public void sensorEvent() {
		// TODO: Receives an updated Model when a new sensor event occurs
	}
	
	public static void main(String[] args) {	// PROGRAM LAUNCH
		new Main();
	}
	
}
