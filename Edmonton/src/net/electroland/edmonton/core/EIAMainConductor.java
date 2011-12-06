package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.awt.Rectangle;
import java.io.IOException;
import java.util.Hashtable;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.ClipEvent;
import net.electroland.ea.ClipListener;
import net.electroland.edmonton.test.TestModel;
import net.electroland.eio.IOManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor extends Thread implements ClipListener {

	static Logger logger = Logger.getLogger(EIAMainConductor.class);

	private ELUManager elu;
	private ELUCanvas2D canvas;
	private IOManager eio;
	private TestModel model;
	private SoundController soundController;
	
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
		context = new Hashtable<String, Object>();
		
		String propsFileName = "EIA.properties";
		logger.info("EIAMain loading " + propsFileName);
        ElectrolandProperties props = new ElectrolandProperties(propsFileName);
        context.put("props",props);
        

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
        context.put("eio",eio);
	    context.put("elu",elu);
        
	    
		canvas = (ELUCanvas2D)elu.getCanvas("EIAspan");
		canvasHeight = canvas.getDimensions().getHeight();
		canvasWidth = canvas.getDimensions().getWidth();
		context.put("canvas",canvas);
		
		
        // create an AnimationManager
        AnimationManager anim = new AnimationManager();
        anim.setContext(context);
        anim.config("EIA-anim.properties");
     // have the frame listen for clip events
        anim.addClipListener(this);
        context.put("animmanager",anim);
        
	    
	   	    
	    soundController = new SoundController(context);
	    context.put("soundController", soundController);


		ef = new EIAFrame(1400,720,context);
		
		
		 // what should this rectangle be defined as, clip size or stage size?
        int clipId0 = anim.startClip("testClip", new Rectangle(0,0,16,16), 1.0);
		

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
	
    @Override
    public void clipEnded(ClipEvent e) {
        logger.info("clip " + e.clipId + " of type " + e.clipId + " ended.");
    }

    @Override
    public void clipStarted(ClipEvent e) {
        logger.info("clip " + e.clipId + " of type " + e.clipId + " started.");
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