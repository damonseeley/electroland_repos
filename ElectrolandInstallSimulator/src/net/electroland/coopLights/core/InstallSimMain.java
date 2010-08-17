package net.electroland.coopLights.core;

import java.awt.geom.Point2D;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.SocketException;
import java.net.UnknownHostException;
import java.util.concurrent.ConcurrentHashMap;
import java.util.Vector;


import net.electroland.installsim.core.Person;
import net.electroland.installsim.sensors.PhotoelectricTripWire;
import net.electroland.udpUtils.UDPLogger;
import net.electroland.installsim.sensors.*;
import net.electroland.installsim.ui.EISFrame;

//import javax.swing.JApplet;

public class InstallSimMain {

	public static Vector Sensors = new Vector ();
	
	// this is eg. of casting Vector to
	public static ConcurrentHashMap<Integer, Person> people = new ConcurrentHashMap<Integer, Person>();
	


	private static InstallSimConductor conductor;
	public static boolean SHOWUI;
	
	public static Integer DUMMYID = -1; 
	
	//UI values
	public static float xScale = 1.0f;
	public static float yScale = xScale;
	public static float xOffset = 80.0f;
	public static float yOffset = 70.0f;


	// use a negative value for mouse control id so as not to be confused with real tracks (which I think are all positive)

	// constructor does all the setup work
	public InstallSimMain() {
		
		SHOWUI = true;

		//logging?
		//initLogging(); 

		initSensors();
		
		initPeople();

		// this puts the menubar in the correct place on macs (at the top of the screen vs at the top of the window) 
		// but won't effect windows.  Its a good habit to call it before you create your frame
		System.setProperty("apple.laf.useScreenMenuBar", "true");
		
		//System.out.println("opening frame");
		new EISFrame();

		// create the conductor thread and get it going
		conductor = new InstallSimConductor(EISFrame.eISPanel);
		conductor.start();

	}
	
	
	private void initSensors() {
		//for now setup specific 
		float startx = 70;
		float starty = 50;
		float incy = 20;
		int[] vec = {45,0,0};
		for (int i=0; i<27; i++) {
			PhotoelectricTripWire s = new PhotoelectricTripWire(i,startx,starty+incy*i,0,vec);
			Sensors.add(s);
		}
		System.out.println(Sensors.toString());
		
	}
	
	private void initPeople() {
		
		
	}

	public void initLogging() {
		try {
			UDPLogger logger = new UDPLogger("120406_03.txt", 6000);
			logger.start(); // changed from start to run EGM
		} catch (IOException e) {
			e.printStackTrace();
		}
	}


	
	public static void killTheads() {
		conductor.stopRunning();	
	}


	

	
	public static void executeRestart() {
		conductor.stopRunning();
		conductor = null;
	}
	
	public static void restartConductor() {
		conductor = new InstallSimConductor(EISFrame.eISPanel);
		conductor.start();

	}
	
	public static void shutdown() {
		try { // surround w/try catch block to make sure System.exit(0) gets call no matter what
			InstallSimMain.killTheads();
		} catch (Exception e) {
			e.printStackTrace();
		}
		try{ // split try/catch so if there is a problem killing threads lights will still die
			//CoopLightsMain.killLights();
		} catch (Exception e) {
			e.printStackTrace();
		}
		System.exit(0);	
	}
	

	
	public static void createTestPerson(){
		// Add a test person
		Person newPerson = new Person(DUMMYID, -500, -500, 0);
		people.put(DUMMYID, newPerson);
	}
	
	public static void main(String[] args) {
		new InstallSimMain();
	}
	
}
