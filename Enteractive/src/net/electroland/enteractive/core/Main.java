package net.electroland.enteractive.core;

import processing.core.PApplet;
import processing.core.PConstants;
import net.electroland.animation.Animation;
import net.electroland.animation.AnimationListener;
import net.electroland.animation.AnimationThread;
import net.electroland.animation.Raster;
import net.electroland.enteractive.gui.GUI;
import net.electroland.enteractive.gui.GUIWindow;
import net.electroland.enteractive.scheduler.TimedEvent;
import net.electroland.enteractive.scheduler.TimedEventListener;
import net.electroland.enteractive.shows.ExampleAnimation;

/**
 * Initiates the program and controls show changes
 * @author asiegel
 */

public class Main implements TimedEventListener, SensorListener, AnimationListener {
	// TODO: This class must implement ShowThreadListener

	SoundManager soundManager;
	PApplet p5;
	AnimationThread currentAnimation;
	GUIWindow guiWindow;
	
	public Main(){
		loadProperties();
		soundManager = new SoundManager(2,2,null);
		// TODO: buffer sound files from properties
		// TODO: Start sensorManager
		
		// start gui
		guiWindow = new GUIWindow(312, 250);
		guiWindow.setVisible(true);

		/** this is just for testing the new animation classes **/
		// should this raster be used across all Animations?
		Raster raster = new Raster(guiWindow.gui.createGraphics(16,11,PConstants.P3D));
		
		// if it is used across all animations, can we send a reference to the GUI for visualizing?
		((GUI)guiWindow.gui).setRaster(raster);
		
		// should this AnimationThread be used for all Animations?
		Animation test = (Animation)new ExampleAnimation(new Model(), raster, soundManager);
		currentAnimation = new AnimationThread(test, 30);
		currentAnimation.start();
	}
	
	private void loadProperties(){
		// TODO: Load properties from file
	}
	

	public void animationComplete(Animation a) {
		// TODO Auto-generated method stub
		
	}

	public void timedEvent(TimedEvent event) {
		// TODO: Trigger scheduled show changes here
	}

	public void sensorEvent() {
		// TODO: Receives an updated Model when a new sensor event occurs
	}
	
	public static void main(String[] args) {	// PROGRAM LAUNCH
		new Main();
	}
	
}
