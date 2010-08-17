package net.electroland.coopLights.core;

import java.util.Enumeration;
import java.util.Random;


import net.electroland.installsim.core.Timer;
import net.electroland.installsim.ui.EISPanel;

public class InstallSimConductor extends Thread { // was implements Runnable.  Extends thread is clearer.
	// You only need implements runnable when you are extending class that doesn't derive from class (so you can't extend thread)

	boolean isRunning;

	public EISPanel ui;

	private float framerate;

	Timer timer;

	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place

	
	//constructor
	public InstallSimConductor(EISPanel theUI) {
		ui = theUI;
		framerate = 40;
		isRunning = true;

		timer = new Timer(framerate);

	}

	public void run() {
		
		timer.start();
		
		curTime = System.currentTimeMillis();
		
		while (isRunning) {
			long tmpTime = System.currentTimeMillis();
			elapsedTime = tmpTime - curTime;
			curTime = tmpTime;


			//detectCollisionsAndTripWires();

			//update people locations
			
			//detect sensor states
			
			//broadcast sensor states


			//paint people and sensors
			if (InstallSimMain.SHOWUI){
				ui.repaint();
			}

			timer.block();

		}

	}
	
	public static int peopleCount() {
		int peopleCount =  0;
		peopleCount = InstallSimMain.people.size();
		return peopleCount;
	}


	public void stopRunning() { // it is good to have a way to stop a thread explicitly (besides System.exit(0) ) EGM
		isRunning = false;
		timer.stopRunning();
	}
	
	public void restart() {
		isRunning = true;
		timer.start();
	}
	


}
