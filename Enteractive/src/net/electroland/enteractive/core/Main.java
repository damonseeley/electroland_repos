package net.electroland.enteractive.core;

import processing.core.PApplet;
import processing.core.PConstants;
import net.electroland.animation.Animation;
import net.electroland.animation.AnimationListener;
import net.electroland.animation.AnimationThread;
import net.electroland.animation.Raster;
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
		guiWindow = new GUIWindow(300, 150);
		guiWindow.setVisible(true);
		
		p5 = new PApplet();		// Processing used for generating raster
		p5.noLoop();

		/** this is just for testing the new animation classes **/
		// should this raster be used across all Animations?
		Raster raster = new Raster(p5.createGraphics(11,16,PConstants.P3D));
		Animation test = (Animation)new ExampleAnimation(new Model(), raster, soundManager);
		// should this AnimationThread be used for all Animations?
		currentAnimation = new AnimationThread(test, 30);
		currentAnimation.start();
	}
	
	private void loadProperties(){
		// TODO: Load properties from file
	}
	

	@Override
	public void animationComplete(Animation a) {
		// TODO Auto-generated method stub
		
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
