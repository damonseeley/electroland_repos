package net.electroland.connection.core;

import org.apache.log4j.Logger;

import net.electroland.connection.util.NoDataException;
import net.electroland.connection.util.RunningAverage;

/**
 * Adapted from NoHo MainFrame.java authored by Eitan Mendelowitz.
 * Revised by Aaron Siegel
 */

public class RenderThread extends Thread {

	static Logger logger = Logger.getLogger(RenderThread.class);
	static Logger peopleCounter = Logger.getLogger("PeopleCount");

	boolean isRunning;
	boolean controlsOn;
	long  ticksPerFrame;
	long lastExitAvg;
	public LightController lightController;	// controls communication with lighting hardware
	public Conductor conductor;				// controls animation transitions
	public Light[] lights;						// grid of lights
	public byte[] buffer;
	
	public RenderThread(float fps) {
		lightController = new LightController(6,28);
		lights = new Light[6*28];							// empty array
		int count = 0;										// count x and y
		for(int y=0; y<28; y++){							// for each y position
			for(int x = 0; x<6; x++){						// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				count++;
			}
		}
		buffer = new byte[lights.length*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		conductor = new Conductor(lights);
		ticksPerFrame = (long) (1000.0f / fps); 
		logger.info("Rendering at " + fps + " fps");
		isRunning = true;
		controlsOn = true;
		lastExitAvg = System.currentTimeMillis();
	}
	
	public byte[] getBuffer(){
		return buffer;
	}
	
	public void run() {
		boolean notWarned = false;
		
		long startTime;
		long lastTime;
		long dTime, lastPeopleCountCheckTime;

		// defaults for running average of people counter
		int samples = Integer.parseInt(ConnectionMain.properties.get("PeopleCount.CountSamples"));
		long sampleRate = Integer.parseInt(ConnectionMain.properties.get("PeopleCount.CountSampleRate"))*1000L;
		long reportRate = Integer.parseInt(ConnectionMain.properties.get("PeopleCount.ReportRate"))*1000L;
		
		// This will store and calculate running averages.
		RunningAverage average = new RunningAverage(samples);
		
		lastTime = System.currentTimeMillis();
		lastPeopleCountCheckTime = lastTime;
		while(isRunning) {
			
			startTime = System.currentTimeMillis();
			average.addValue((double)ConnectionMain.personTracker.peopleCount(), sampleRate);

			if (startTime - lastPeopleCountCheckTime > reportRate){
				// report the number of people in the room once a minute.
				// good time to revise duration of tracking/screen saver modes
				try {
					int peoplecount = (int)average.getAvg();
					peopleCounter.info(peoplecount);
					/*
					// this was moved inside the conductor, since it uses current population NOT average
					trackingduration = peoplecount*4000 + 11000;		// SET MOVING TRACKING DURATION HERE
					if(trackingduration > 180000){						// no more than 3 minutes long
						trackingduration = 180000;
					}
					conductor.trackingConnections.setDefaultDuration(trackingduration);
					conductor.screenSaver.setDefaultDuration(trackingduration);
					*/
					if(peoplecount > 0 && peoplecount <= 5){			// CONDITIONS FOR LITTLE SHOWS
						conductor.littleShows = true;
					} else {
						conductor.littleShows = false;
					}
				} catch (NoDataException e) {
					logger.error("Error printing out average.", e);
					peopleCounter.error("Error printing out average.", e);
				}
				lastPeopleCountCheckTime = startTime;
			}
			
			buffer = conductor.draw();										// controls all transitions between display modes
			if(startTime - lastTime > 5000){								// every 5 seconds...
				lightController.updateLightBoxes(buffer);
				lastTime = System.currentTimeMillis();
			} else {
				lightController.updateLights(buffer);						// lightController sends byte string
			}
			
			if(System.currentTimeMillis() - lastExitAvg > 5000){
				ConnectionMain.personTracker.exitavg = ConnectionMain.personTracker.exitcounter / (float)5;
				ConnectionMain.personTracker.exitcounter = 0;
				//System.out.println(ConnectionMain.personTracker.exitavg +" exits per second");
				lastExitAvg = System.currentTimeMillis();
			}
			
			dTime = startTime + ticksPerFrame - System.currentTimeMillis();	// set delay remainder for frame
			
			if(dTime > 0) {
				notWarned = true;
				try {
					sleep(dTime);
				} catch (InterruptedException e) {
				}
			} else {
				if(notWarned) {
					logger.debug("Warning: framerate falling behind");
					notWarned = false;
				}
			}
			//lastTime = startTime;
		}
	}

}
