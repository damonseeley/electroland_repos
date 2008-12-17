package net.electroland.connection.core;

import net.electroland.connection.animation.Animation;
import net.electroland.connection.animation.Biggest;
import net.electroland.connection.animation.Matrix;
import net.electroland.connection.animation.MusicBox;
import net.electroland.connection.animation.TrackingConnections;
import net.electroland.connection.animation.TrackingMain;
import net.electroland.connection.animation.Transition;
import net.electroland.connection.animation.TransitionListener;
import net.electroland.connection.animation.Vegas;
import net.electroland.connection.animation.Wave;
import net.electroland.connection.animation.ScreenSaver;
import net.electroland.connection.animation.QuickColor;
import net.electroland.connection.animation.RandomFill;
import net.electroland.connection.animation.transitions.FadeTransition;

/**
 * This handles the transitions between display modes. and passes the buffer back to LightController.
 */

public class Conductor implements TransitionListener{
	
	public int gridx = 6;
	public int gridy = 28;
	public Light[] lights;							// light objects
	public byte[] buffer, oldbuffer;				// output data buffer
	public boolean connectionMode;				// trackingmain or trackingconnections
	public int lastMode;							// previous animation mode
	public int nextMode;							// next show to play
	public long lastChange;						// system clock in milliseconds
	public boolean fadeOn;						// activates buffer blending
	public float crossFade;						// fade value between display modes
	public float crossFadeSpeed;					// speed of cross fade
	public boolean transition = Boolean.parseBoolean(ConnectionMain.properties.get("transition"));
	public boolean littleShows = false;
	public boolean automatedSwitching = true;
	private int trackingduration;					// tracking/screensaver duration in milliseconds
	
	// FULL SHOWS
	public Vegas vegaspulse;
	public Wave wave;
	public TrackingMain trackingBars;
	public TrackingConnections trackingConnections;
	public MusicBox musicbox;
	public Matrix matrix;
	public Biggest biggest;
	public ScreenSaver screenSaver;
	
	// LITTLE SHOWS
	public RandomFill randomfill;
	public QuickColor blue;
	public QuickColor red;
	public QuickColor purple;
	public Wave littlewave;
	public Matrix littlematrix;

	// this is the currently "live" animation
	private Animation currentAnimation;
	private int currentShowIndex, littleShowIndex;
	private Animation[] shows;
	private Animation[] littleshowlist;
	
	public Conductor(Light[] _lights){

		lights = _lights;

		// FULL SHOWS + TRACKING/SCREEN SAVER
		vegaspulse = new Vegas(gridx, gridy, true, getIntProp("vegasDuration"));
		wave = new Wave(gridx, gridy, getIntProp("waveDuration"), ConnectionMain.properties.get("soundWaveGlobal"));
		trackingConnections = new TrackingConnections(lights, getIntProp("trackingDuration"));
		trackingBars = new TrackingMain(lights, getIntProp("trackingDuration"));
		musicbox = new MusicBox(lights, getIntProp("musicboxDuration"));
		matrix = new Matrix(lights, getIntProp("matrixDuration"), ConnectionMain.properties.get("soundMatrixGlobal"));
		biggest = new Biggest(lights, 120, 31, getIntProp("biggestDuration"));
		screenSaver = new ScreenSaver(getIntProp("trackingDuration"), 30, 3000, 1000, 500);

		Animation[] shows = {vegaspulse, wave, musicbox, matrix, biggest};
		//Animation[] shows = {screenSaver};

		// LITTLE SHOWS
		randomfill = new RandomFill(4000);
		blue = new QuickColor(5500, 0, 255, ConnectionMain.properties.get("soundQuickBlue"));
		red = new QuickColor(5500, 255, 0, ConnectionMain.properties.get("soundQuickRed"));
		purple = new QuickColor(5500, 255, 255, ConnectionMain.properties.get("soundQuickPurple"));
		littlewave = new Wave(gridx, gridy, 5000, ConnectionMain.properties.get("soundLittleWave"));
		littlematrix = new Matrix(lights, 4000, ConnectionMain.properties.get("soundLittleMatrix"));
		Animation[] littleshowlist = {randomfill, blue, red, purple, littlewave, littlematrix};
		
		this.shows = shows;
		this.littleshowlist = littleshowlist;

		// state 
		currentShowIndex = 0;
		littleShowIndex = 0;
		connectionMode = false;		// defaults to bar mode
		if(connectionMode){
			currentAnimation = trackingConnections;
		} else {
			currentAnimation = trackingBars;
		}
		lastChange = System.currentTimeMillis();
	}
	
	public byte[] draw(){

//		if (currentAnimation instanceof Transition){
//			// no time duration check if we are Transitioning.
//			// (that means that our shows are actually show duration + the transition duration)
//			lastChange = System.currentTimeMillis();
//			
//		}else 
		if(automatedSwitching){		// this gets disabled when shows are manually selected from the GUI
			if (System.currentTimeMillis() - lastChange > currentAnimation.getDefaultDuration()){
	
				Animation nextAnimation;
	
				if (currentAnimation instanceof TrackingConnections || currentAnimation instanceof TrackingMain || currentAnimation instanceof ScreenSaver){
				//if (currentAnimation instanceof TrackingConnections){
					if(littleShows){	// LITTLE ANIMATION MODE
						nextAnimation = littleshowlist[littleShowIndex++];
						if (littleShowIndex == littleshowlist.length){
							littleShowIndex = 0;
						}
					} else {			// BIG ANIMATION MODE
						nextAnimation= shows[currentShowIndex++];
						if(ConnectionMain.personTracker.peopleCount() == 0 && nextAnimation instanceof MusicBox){
							nextAnimation= shows[currentShowIndex++];	// skip music box when nobody is around
						}
						if (currentShowIndex == shows.length){
							currentShowIndex = 0;
						}
					}
				}else{
					if(ConnectionMain.personTracker.peopleCount() > 0){
						if(connectionMode){
							nextAnimation = trackingConnections;
						} else {
							nextAnimation = trackingBars;
						}
					} else {
						screenSaver.setDefaultDuration(300000);	// 5 minutes of screen saver between shows
						nextAnimation = screenSaver;
						ConnectionMain.soundController.killAllSounds();	// kill all sounds in MAX/MSP
					}
				}
	
				nextAnimation.start();
				currentAnimation = new FadeTransition(currentAnimation, nextAnimation);
				((Transition)currentAnimation).addListener(this);
				lastChange = System.currentTimeMillis();
			} else {
				// screen saver code is going to be something like:		
				if (ConnectionMain.personTracker.peopleCount() == 0 && currentAnimation instanceof TrackingConnections){
					currentAnimation = new FadeTransition(currentAnimation, screenSaver);
					((Transition)currentAnimation).addListener(this);
				} else if(ConnectionMain.personTracker.peopleCount() > 0 && currentAnimation instanceof ScreenSaver){
					if(connectionMode){
						currentAnimation = new FadeTransition(currentAnimation, trackingConnections);
					} else {
						currentAnimation = new FadeTransition(currentAnimation, trackingBars);
					}
					((Transition)currentAnimation).addListener(this);
					lastChange += 15000;	// current time + 15 seconds
				}
				trackingduration = ConnectionMain.personTracker.peopleCount()*5000 + 25000;		// SET MOVING TRACKING DURATION HERE
				if(trackingduration > 180000){						// no more than 3 minutes long
					trackingduration = 180000;
				}
				trackingConnections.setDefaultDuration(trackingduration);
				screenSaver.setDefaultDuration(trackingduration);
			}
		}
		
		return currentAnimation.draw();
	}
	
	public int getTrackingDuration(){
		return trackingduration;
	}
	
	public void setMode(int mode){
		if(mode == -1){
			if(connectionMode){
				currentAnimation = trackingConnections;
			} else {
				currentAnimation = trackingBars;
			}
			automatedSwitching = true;
		} else if(mode == -2){
			currentAnimation = screenSaver;
			currentAnimation.start();
			automatedSwitching = false;		// disabled automated switching for each documentation
		} else {
			currentAnimation = shows[mode];
			currentAnimation.start();
			automatedSwitching = false;		// disabled automated switching for each documentation
		}
	}
	
	public void playLittleShow(int mode){
		// used only for demo purposes
		// plays a quick show and returns to tracking mode
		Animation nextAnimation = littleshowlist[mode];
		nextAnimation.start();
		currentAnimation = new FadeTransition(currentAnimation, nextAnimation);
		((Transition)currentAnimation).addListener(this);
		lastChange = System.currentTimeMillis();
	}

	public void transitionComplete(Transition transition) {
		// when a transition is complete, the transition will call you.
		// let's just assume you should set the current Animation to whatever
		// the animation ended on.
		currentAnimation = transition.getEndAnimation();
	}

	// half-assed utility to reduce verbosity.  should probable ACTUALLY
	// check to see if the prop exists here, and handle number format exceptions.
	private static int getIntProp(String name){
		return Integer.parseInt(ConnectionMain.properties.get(name));
	}

}
