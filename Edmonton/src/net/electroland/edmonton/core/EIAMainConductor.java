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
import net.electroland.edmonton.core.model.OneEventPerPeriodModelWatcher;
import net.electroland.eio.IOManager;
import net.electroland.eio.IOState;
import net.electroland.eio.IState;
import net.electroland.eio.model.Model;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.eio.model.ModelWatcher;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor extends Thread implements ClipListener, ActionListener, ModelListener {

	static Logger logger = Logger.getLogger(EIAMainConductor.class);

	private ElectrolandProperties props;
	private ELUManager elu;
	private ELUCanvas2D canvas;
	private IOManager eio;
	private Model model;
	private SoundController soundController;
	private AnimationManager anim;

	public int canvasHeight, canvasWidth;
	public Hashtable<String, Object> context;

	private Timer startupTestTimer, timedShows;

	public EIAFrame ef;
	
	private ModelWatcher test;

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


		model = new Model();
		test = new OneEventPerPeriodModelWatcher(500);
		IOState i30 = eio.getStateById("i30");
		//logger.info(i30);
		//logger.info("Got IOState " + i30.getID() + " location " + i30.getLocation());
	//	Collection<IState> st = Collection<IState>eio.getStates();
		model.addModelWatcher(test, "testWatcher", eio.getIStates());
		model.addModelListener(this);
		

		ef = new EIAFrame(Integer.parseInt(props.getRequired("settings", "global", "guiwidth")),Integer.parseInt(props.getRequired("settings", "global", "guiheight")),context);

		// Thread setup
		framerate = props.getRequiredInt("settings", "global", "framerate");

		isRunning = true;
		timer = new FrameTimer(framerate);
		start();
		logger.info("EIA started up at framerate = " + framerate);

		startupTestTimer = new Timer();
		startupTestTimer.schedule(new startupTests(), 4000);
		
		timedShows = new Timer();
		timedShows.schedule(new timedShowPlayer(),60000);


		ef.addButtonListener(this);
	}
	
	
	
	/************************* Model Handlers ******************************/


	@Override
	public void modelChanged(ModelEvent evt) {
		logger.info("Model Event: " + evt.getSource());
		// TODO Auto-generated method stub

	}

	
	
	/************************* Test Event Handlers ******************************/

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
		if ("egg1".equals(e.getActionCommand())) {
			Egg1(200.0);
		}

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

	
	
	/************************* Animations ******************************/


	class timedShowPlayer extends TimerTask {
		public void run() {
			BigFill();
			//play again in 60s
			timedShows.schedule(new timedShowPlayer(), 60000);
			
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
		//soundController.playSingleBay("test_1.wav", 600.0, 1.0f); // plays a sound out of the speaker nearest to the x value provided

		// TEST CLIP
		// create clip off stage left
		int clip = anim.startClip("testClip", new Rectangle(-14,0,16,16), 1.0);
		// expand clip1 to full screen
		anim.queueClipChange(clip, new Rectangle(0,0,(int)canvasWidth,(int)canvasHeight), null, null, 2000, 0, false);
		// retract clip to right
		anim.queueClipChange(clip, new Rectangle(canvasWidth,0,16,16), null, null, 3000, 0, true);
	}

	private void Egg1(double x) {
		int startWidth = 2;
		int endWidth = 48;
		int clip = anim.startClip("imageClipNoise", new Rectangle((int)x - startWidth/2,0,startWidth,16), 1.0);
		
		anim.queueClipChange(clip, new Rectangle((int)x - endWidth/2,0,endWidth,16), null, null, 1000, 0, false);
		anim.queueClipChange(clip, new Rectangle((int)x - startWidth/2,0,startWidth,16), null, null, 500, 0, true);

	}







	
	/************************* Main Loop ******************************/
	
	public void run() {
		timer.start();
		curTime = System.currentTimeMillis();

		while (isRunning) {
			/*
			 * DO STUFF
			 */

			model.poll();

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