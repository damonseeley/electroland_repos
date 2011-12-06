package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.io.IOException;
import java.util.Hashtable;

import net.electroland.ea.AnimationManager;
import net.electroland.edmonton.test.TestModel;
import net.electroland.eio.IOManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor extends Thread {

	static Logger logger = Logger.getLogger(EIAMainConductor.class);

	private ELUManager elu;
	private ELUCanvas2D canvas;
	private IOManager eio;
	private TestModel model;
	
	public double canvasHeight, canvasWidth;
	public Hashtable<String, Object> context;
	
	//private SoundManager soundManager;

	
	public EIAFrame ef;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static Timer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place


	public EIAMainConductor()
	{
		String propsFileName = "EIA.properties";
		logger.info("EIAMain loading " + propsFileName);
        ElectrolandProperties props = new ElectrolandProperties(propsFileName);

		elu = new ELUManager();
        eio = new IOManager();
        try {
            elu.load("EIA-ELU.properties");
            eio.load("EIA-EIO.properties");
            eio.start();
        } catch (OptionException e) {
            e.printStackTrace();
            System.exit(0);
        } catch (IOException e) {
            e.printStackTrace();
            System.exit(0);
        }
        
		canvas = (ELUCanvas2D)elu.getCanvas("EIAspan");
		canvasHeight = canvas.getDimensions().getHeight();
		canvasWidth = canvas.getDimensions().getWidth();
		
		
		
        // create an AnimationManager
        AnimationManager anim = new AnimationManager();
        anim.setContext(context);
        anim.config("EIA-anim.properties");
        
        
        
				
	    context = new Hashtable<String, Object>();
	    //context.put("sound_manager", soundManager)
	    context.put("eio",eio);
	    context.put("elu",elu);
	    context.put("canvas",canvas);
	    context.put("animmanager",anim);
	    context.put("props",props);
	    //create reserved names 


		ef = new EIAFrame(1400,720,context);
		

		////// THREAD SETUP
		framerate = 30;
		
		isRunning = true;
		timer = new Timer(framerate);
		start();
		logger.info("EIA started up at framerate = " + framerate);

	}

	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {

			/*
			 * DO STUFF
			 */
			
			
			
			
			// Update the GUI Panel
			ef.update();

			//Thread ops
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

	public static void main(String args[])
	{
		new EIAMainConductor();

	}




}