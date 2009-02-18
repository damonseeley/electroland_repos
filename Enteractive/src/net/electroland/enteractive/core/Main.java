package net.electroland.enteractive.core;

import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;

/**
 * ENTERACTIVE by Electroland - Spring 2009
 * @author asiegel
 */

public class Main implements TimedEventListener, SensorListener {
	// TODO: This class must implement ShowThreadListener

	SoundManager soundManager;
	
	public Main(){
		loadProperties();
		soundManager = new SoundManager(2,2,null);
		// TODO: buffer sound files from properties
		// TODO: Start sensor manager
	}
	
	private void loadProperties(){
		// TODO: Load properties from file
	}

	@Override
	public void timedEvent(TimedEvent event) {
		// TODO Trigger scheduled show changes here
	}

	@Override
	public void sensorEvent() {
		// TODO Trigger easter-egg shows based on special circumstances/patterns
	}
	
	public static void main(String[] args) {	// PROGRAM LAUNCH
		new Main();
	}
	
}
