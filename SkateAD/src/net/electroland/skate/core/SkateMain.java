package net.electroland.skate.core;

import java.awt.Color;
import java.awt.Graphics;
import java.awt.color.ColorSpace;
import java.awt.image.BufferedImage;
import java.awt.image.ImageObserver;

import net.electroland.skate.ui.GUIFrame;
import net.electroland.skate.ui.GUIPanel;
import net.electroland.utils.lighting.ElectrolandLightingManager;

import org.apache.log4j.Logger;


/**
 * @title	"SKATE 1.0" by Electroland, A+D Summer 2011
 * @author	Damon Seeley & Bradley Geilfuss
 */

public class SkateMain extends Thread {

	private static Logger logger = Logger.getLogger(SkateMain.class);

	private ElectrolandLightingManager elu;
	//private Canvas2D c;
	
	public static boolean SHOWUI;
	public static GUIFrame guiFrame;
	public static GUIPanel guiPanel;
	
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place


	public SkateMain() {
		
		// create lighting utils
		//elu = new ElectrolandLightingManger("lights.properties");
		//c = elu.getCanvas();

		
		SHOWUI = true;
		int GUIWidth = 800;
		int GUIHeight = 800;
		guiFrame = new GUIFrame(GUIWidth,GUIHeight);
		guiPanel = new GUIPanel(GUIWidth,GUIHeight);
		//add the panel to the top of the window
		guiFrame.add(guiPanel);
		
		

		//init space and lights
		//init skaters
		
		//init sound controller and speakers
		
		
		// start everything (e.g., start the threads for each of these subsystems)
		//elu.start();
		
		
		// thread stuff
		framerate = 30;
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("Skate 1.0 started up");


	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {

			//test
			//BufferedImage i = new BufferedImage(400,400,ColorSpace.TYPE_RGB);
			Graphics g = guiPanel.getGraphics();
			//g.setColor(new Color(255,0,0));
			//g.fillRect((int)(20),(int)(20),20,20);
			//logger.info("Drew something?");
			
			BufferedImage i = new BufferedImage(400,400,ColorSpace.TYPE_RGB);
			Graphics gi = i.getGraphics();
			gi.setColor(new Color(255,150,255));
			gi.fillRect(0,0,i.getWidth(),i.getHeight());
			gi.setColor(new Color(255,0,0));
			gi.fillRect((int)(60),(int)(60),20,20);
			
			g.drawImage(i, 0, 0, null);
			
			
			
			
			
			//figure out whether to add or subtract skaters
			
			//update sound locations
			
			//draw skater sprites on an image at native size
			//flop sand scale skater image to canvas-size
			//extract a pixel array from the canvas-sized sprite image and sync with ELU
			//draw detectors and skater info on the local canvas image post sync
			
			
			//logger.info(timer.sleepTime);
			timer.block();
		}



	}

	
	public static void killTheads() {
		stopRunning();	
	}
	
	public static void stopRunning() { // it is good to have a way to stop a thread explicitly (besides System.exit(0) ) EGM
		isRunning = false;
		timer.stopRunning();
	}
	
	public static void restart() {
		isRunning = true;
		timer.start();
	}
	
	public static void shutdown() {
		try { // surround w/try catch block to make sure System.exit(0) gets call no matter what
			SkateMain.killTheads();
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


	public static void main(String[] args){
		new SkateMain();
	}


}


