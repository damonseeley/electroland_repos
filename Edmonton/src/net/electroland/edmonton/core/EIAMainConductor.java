package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.awt.Rectangle;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Timer;
import java.util.TimerTask;

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

public class EIAMainConductor extends Thread implements ClipListener, ActionListener {

	static Logger logger = Logger.getLogger(EIAMainConductor.class);

	private ElectrolandProperties props;
	private ELUManager elu;
	private ELUCanvas2D canvas;
	private IOManager eio;
	private TestModel model;
	private SoundController soundController;
	private AnimationManager anim;

	public int canvasHeight, canvasWidth;
	public Hashtable<String, Object> context;

	private Timer startupTestTimer;

	public EIAFrame ef;

	//Thread stuff
	public static boolean isRunning;
	private static float framerate;
	private static FrameTimer timer;
	public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
	public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place


	public EIAMainConductor()
	{
		context = new Hashtable<String, Object>();

		String propsFileName = "EIA.properties";
		logger.info("EIAMain loading " + propsFileName);
		props = new ElectrolandProperties(propsFileName);
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

		/*
		 * create model and add watchers and listeners
		 */

		canvas = (ELUCanvas2D)elu.getCanvas("EIAspan");
		canvasHeight = (int)canvas.getDimensions().getHeight();
		canvasWidth = (int)canvas.getDimensions().getWidth();
		context.put("canvas",canvas);

		// create an AnimationManager
		anim = new AnimationManager();
		anim.setContext(context);
		anim.config("EIA-anim.properties");
		// listen for clip events
		anim.addClipListener(this);
		context.put("anim",anim);
		context.put("animpropsfile", "EIA-anim.properties");

		soundController = new SoundController(context);
		context.put("soundController", soundController);



		ef = new EIAFrame(Integer.parseInt(props.getRequired("settings", "global", "guiwidth")),Integer.parseInt(props.getRequired("settings", "global", "guiheight")),context);

		// Thread setup
		framerate = props.getRequiredInt("settings", "global", "framerate");

		isRunning = true;
		timer = new FrameTimer(framerate);
		start();
		logger.info("EIA started up at framerate = " + framerate);

		startupTestTimer = new Timer();
		startupTestTimer.schedule(new startupTests(), 4000);

		ef.addButtonListener(this);
	}

	// handle the actions from test buttons
	public void actionPerformed(ActionEvent e) {
		logger.info(e.getActionCommand());
		if ("shooter1".equals(e.getActionCommand())) {
			Shooter1();
		}
		if ("bigfill".equals(e.getActionCommand())) {
			BigFill();
		}
		if ("shooter2".equals(e.getActionCommand())) {
			Shooter2();
		}

	}

	private void Shooter1() {

		int clip = anim.startClip("testClip", new Rectangle(canvasWidth-2,0,16,16), 1.0);
		anim.queueClipChange(clip, new Rectangle(280,0,48,16), null, 0.0, 1400, 0, true);
	}

	private void Shooter2() {

		int clip = anim.startClip("testClip", new Rectangle(250,0,16,16), 1.0);
		anim.queueClipChange(clip, new Rectangle(-50,0,48,16), null, 0.0, 1200, 0, true);
	}

	private void BigFill() {

		soundController.playSingleBay("test_1.wav", 600.0, 1.0f); // plays a sound out of the speaker nearest to the x value provided

		// TEST CLIP
		// create clip off stage left
		int clip = anim.startClip("testClip", new Rectangle(-14,0,16,16), 1.0);

		// expand clip1 to full screen
		anim.queueClipChange(clip, new Rectangle(0,0,(int)canvasWidth,(int)canvasHeight), null, null, 2000, 0, false);

		// retract clip to right
		anim.queueClipChange(clip, new Rectangle(canvasWidth,0,16,16), null, null, 3000, 0, true);

	}

	private void Egg1() {

	}


	class startupTests extends TimerTask {
		public void run() {
			//startupTestTimer

			// TEST SOUND
			//soundController.playTestSound("test_1.wav");
			//soundController.playSingleBay("test_1.wav", 600.0, 1.0f); // plays a sound out of the speaker nearest to the x value provided

			//startupTestTimer.schedule(new startupTests(), 10000);
		}
	}



	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {
			/*
			 * DO STUFF
			 */

			//model.poll();

			//sync the animMgr to ELU here
			//elu.sync()

			// Update the GUI Panel
			ef.update();

			//Thread ops
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