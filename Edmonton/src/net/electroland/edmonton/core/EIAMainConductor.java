package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Timer;
import java.util.TimerTask;

import net.electroland.ea.AnimationManager;
import net.electroland.edmonton.core.model.LastTrippedModelWatcher;
import net.electroland.edmonton.core.model.ScreenSaverModelWatcher;
import net.electroland.edmonton.core.model.TrackerBasicModelWatcher;
import net.electroland.edmonton.core.sequencing.SimpleSequencer;
import net.electroland.edmonton.core.ui.EIAFrame;
import net.electroland.eio.IOManager;
import net.electroland.eio.model.Model;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.eio.model.ModelWatcher;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor extends Thread implements ActionListener, ModelListener {

    static Logger logger = Logger.getLogger(EIAMainConductor.class);

    private int fadeDuration = 2000;
    private int inactivityThreshold = 1000 * 10;
    private ElectrolandProperties props;
    private ELUManager elu;
    private boolean updateLighting = true;
    private boolean track;
    private boolean screensaver = true;
    private ELUCanvas2D canvas;
    private IOManager eio;

    private SoundController soundController;
    private AnimationManager anim;
    private SimpleSequencer sequencer;
    private EIAClipPlayer clipPlayer;

    public int canvasHeight, canvasWidth;
    public Hashtable<String, Object> context;

    private Timer startupTestTimer, timedShows;

    public EIAFrame ef;

    private Model model;
    private ModelWatcher stateToBright,entry1,exit1,entry2,exit2,egg1,egg2,egg3,egg4;
    private TrackerBasicModelWatcher tracker;
    private LastTrippedModelWatcher tripRecord;
    private ScreenSaverModelWatcher screenSaver;

    private int stateToBrightnessClip;

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

        boolean eioplayback = false;
        try {
            eioplayback = Boolean.parseBoolean(props.getOptional("settings", "eiomode", "playback"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            eioplayback = false;
            e.printStackTrace();
        }

        try {
            elu.load("EIA-ELU.properties");
            if (eioplayback){
                eio.load("EIA-EIO-playback.properties");
            } else {
                eio.load("EIA-EIO.properties");
            }
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

        updateLighting = Boolean.parseBoolean(props.getOptional("settings", "global", "updateLighting"));
        try {
            track = Boolean.parseBoolean(props.getOptional("settings", "tracking", "track"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            track = false;
            e.printStackTrace();
        }
        try {
            screensaver = Boolean.parseBoolean(props.getOptional("settings", "sequencing", "screensaver"));
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            screensaver = true;
            e.printStackTrace();
        }



        canvas = (ELUCanvas2D)elu.getCanvas("EIAspan");
        canvasHeight = (int)canvas.getDimensions().getHeight();
        canvasWidth = (int)canvas.getDimensions().getWidth();
        context.put("canvas",canvas);

        // create an AnimationManager
        anim = new AnimationManager("EIA-anim.properties");

        context.put("anim",anim);
        context.put("animpropsfile", "EIA-anim.properties");

        soundController = new SoundController(context);
        context.put("soundController", soundController);

        String seqpropsfile = "EIA-seq-LITE.properties";
        sequencer = new SimpleSequencer(seqpropsfile, context);
        context.put("sequencer", sequencer);
        context.put("seqpropsfile", seqpropsfile);

        clipPlayer = new EIAClipPlayer(anim,elu);
        context.put("clipPlayer", clipPlayer);


        /******** Model, Watchers & Timers ********/
        model = new Model();
        model.addModelListener(this);
        model.addModelListener(sequencer);

        createModelWatchers();

        startupTestTimer = new Timer();
        startupTestTimer.schedule(new startupTests(), 3000);

        // disabled for now
        //timedShows = new Timer();
        //timedShows.schedule(new timedShowPlayer(),60000);

        tripRecord = new LastTrippedModelWatcher();
        model.addModelWatcher(tripRecord, "tripRecord", eio.getIStates());

        // watch for screen saver switches
        screenSaver = new ScreenSaverModelWatcher();
        screenSaver.setTimeOut(this.inactivityThreshold);
        model.addModelWatcher(screenSaver,  "screenSaver", eio.getIStates());

        /******** GUI ********/
        ef = new EIAFrame(Integer.parseInt(props.getRequired("settings", "global", "guiwidth")),Integer.parseInt(props.getRequired("settings", "global", "guiheight")),context);
        ef.addButtonListener(this);

        
        //start it all
        //goQuiet();

        /******** Thread Setup ********/
        framerate = props.getRequiredInt("settings", "global", "framerate");
        isRunning = true;
        timer = new FrameTimer(framerate);
        start();
        logger.info("EIA started up at framerate = " + framerate);
    }


    /************************* Test Event Handlers ******************************/

    // handle the actions from test buttons
    public void actionPerformed(ActionEvent e) {
        logger.info(e.getActionCommand());

        if ("startShow1".equals(e.getActionCommand())) {
            this.goLive();
            sequencer.play("show1");
            //this.goLive();
        }
        if ("startShow2".equals(e.getActionCommand())) {
            sequencer.play("show2");
            //this.goLive();
        }
        if ("testShow".equals(e.getActionCommand())) {
            sequencer.play("testShow");
            //this.goLive();
        }
        if ("stopSeq".equals(e.getActionCommand())) {
            sequencer.stop();
            soundController.fadeAll(500);
        }

    }

    class startupTests extends TimerTask {
        public void run() {
            //startupTestTimer
            // TEST SOUND
            //soundController.playTestSound("test_1.wav");
            //soundController.playSingleChannel("test_1.wav", 600.0, 1.0f); // plays a sound out of the speaker nearest to the x value provide
            //startupTestTimer.schedule(new startupTests(), 10000);
            //int faintSparkle = anim.startClip("sparkleClip320", new Rectangle(0,0,635,16), 0.3); // huge sparkly thing over full area
        }
    }

    public void goQuiet()
    {
        if (screensaver){
            sequencer.stop();
            soundController.fadeAll(500);
            sequencer.play(sequencer.quietShowId);
            clipPlayer.live.fadeOut(500).deleteChildren();
            clipPlayer.quiet.fadeIn(0);
        }
    }

    public void goLive(){
        if (screensaver){
            sequencer.stop();
            soundController.fadeAll(500);
            sequencer.play(sequencer.liveShowId);
            clipPlayer.quiet.fadeOut(500).deleteChildren();
            clipPlayer.live.fadeIn(0);
        }

    }

    /************************* Model Handlers ******************************/

    private void createModelWatchers(){

        if (track) {
            tracker = new TrackerBasicModelWatcher(context); //starting with vals of 64 which is what was used in TestConductor (single value)
            model.addModelWatcher(tracker, "tracker", eio.getIStates());
            context.put("tracker", tracker);
        }





        /**
         * disable these events for now to make debugging on future shows easier
         */

        /*
		int stateToBrightnessClip = anim.startClip("stateToBrightnessImage", new Rectangle(0,0,canvasWidth,canvasHeight), 1.0);
		int maxBright = 192; //max brightness for pathtracer
		stateToBright = new StateToBrightnessModelWatcher(16,2,maxBright); //starting with vals of 64 which is what was used in TestConductor (single value)
		model.addModelWatcher(stateToBright, "stateToBright", eio.getIStates());


		entry1 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> entry1states = new ArrayList<IState>();
		entry1states.add(eio.getIStateById("i1"));
		model.addModelWatcher(entry1, "entry1", entry1states);

		exit1 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> exit1states = new ArrayList<IState>();
		exit1states.add(eio.getIStateById("i39"));
		model.addModelWatcher(exit1, "exit1", exit1states);

		entry2 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> entry2states = new ArrayList<IState>();
		entry2states.add(eio.getIStateById("i40"));
		model.addModelWatcher(entry2, "entry2", entry2states);

		exit2 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> exit2states = new ArrayList<IState>();
		exit2states.add(eio.getIStateById("i71"));
		model.addModelWatcher(exit2, "exit2", exit2states);

		egg1 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> egg1states = new ArrayList<IState>();
		egg1states.add(eio.getIStateById("i13"));
		//egg1states.add(eio.getIStateById("i4"));
		model.addModelWatcher(egg1, "egg1", egg1states);

		egg2 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> egg2states = new ArrayList<IState>();
		egg2states.add(eio.getIStateById("i26"));
		model.addModelWatcher(egg2, "egg2", egg2states);

		egg3 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> egg3states = new ArrayList<IState>();
		egg3states.add(eio.getIStateById("i50"));
		model.addModelWatcher(egg3, "egg3", egg3states);

		egg4 = new OneEventPerPeriodModelWatcher(500);
		ArrayList<IState> egg4states = new ArrayList<IState>();
		egg4states.add(eio.getIStateById("i60"));
		model.addModelWatcher(egg4, "egg4", egg4states);
         */
    }


    @Override
    public void modelChanged(ModelEvent evt) {

        if (evt.watcherName == "screenSaver"){
            logger.info("got screen saver event at " + System.currentTimeMillis());
            if (((ScreenSaverModelWatcher)evt.getSource()).isQuiet())
            {
                logger.info("go quiet");
                this.goQuiet();
            }else{
                logger.info("go live");
                this.goLive();
            }
        }





        /*
         * Old show event stuff
         */

        /*
		if (evt.watcherName == "stateToBright"){

			//((StateToBrightnessClip) anim.getClip(stateToBrightnessClip)).setBrightValues(evt.optionalPostiveDetails);
			((StateToBrightnessImageClip) anim.getClip(stateToBrightnessClip)).setBrightValues(evt.optionalPostiveDetails);

		} else if (evt.watcherName == "entry1"){
			entry1Shooter();
		} else if (evt.watcherName == "exit1"){
			exit1Shooter();
		} else if (evt.watcherName == "entry2"){
			entry2Shooter();
		} else if (evt.watcherName == "exit2"){
			exit2Shooter();
		} else if (evt.watcherName == "egg1"){
			ModelWatcher mw = (ModelWatcher)evt.getSource();
			for (Iterator<IState> it = mw.getStates().iterator (); it.hasNext (); ) {
				IState is = (IState)it.next();
				eggSparkle(is.getLocation().x);
			}
		} else if (evt.watcherName == "egg2"){
			ModelWatcher mw = (ModelWatcher)evt.getSource();
			for (Iterator<IState> it = mw.getStates().iterator (); it.hasNext (); ) {
				IState is = (IState)it.next();
				eggExpand(is.getLocation().x);
			}
		} else if (evt.watcherName == "egg3"){
			ModelWatcher mw = (ModelWatcher)evt.getSource();
			for (Iterator<IState> it = mw.getStates().iterator (); it.hasNext (); ) {
				IState is = (IState)it.next();
				eggWave(is.getLocation().x);
			}
		} else if (evt.watcherName == "egg4"){
			ModelWatcher mw = (ModelWatcher)evt.getSource();
			for (Iterator<IState> it = mw.getStates().iterator (); it.hasNext (); ) {
				IState is = (IState)it.next();
				eggSparkle(is.getLocation().x);
			}
		}
         */


    }





    /************************* Animations ******************************/


    /*
     * 
	class timedShowPlayer extends TimerTask {
		public void run() {
			bigFill();
			//play again in 60s
			//timedShows.schedule(new timedShowPlayer(), 60000);
			timedShows.schedule(new timedShowPlayer(), 600000); //set to every 10 minutes during debugging


		}
	}


	int shootEndWidth = 36;
	private void entry1Shooter() {
		int clip = anim.startClip("simpleClip16", new Rectangle(canvasWidth-2,0,16,16), 1.0);
		//anim.queueClipChange(clip, new Rectangle(280,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
		anim.queueClipChange(clip, new Rectangle((canvasWidth-2-280/2),0,shootEndWidth/2+16,16), null, 1.0, 1400, 0, false);
		anim.queueClipChange(clip, new Rectangle(280,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
	}
	private void exit1Shooter() {
		int clip = anim.startClip("simpleClip16", new Rectangle(340,0,16,16), 1.0);
		//anim.queueClipChange(clip, new Rectangle(canvasWidth+16,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
		anim.queueClipChange(clip, new Rectangle(canvasWidth+16-340/2,0,shootEndWidth/2+16,16), null, 1.0, 1400, 0, false);
		anim.queueClipChange(clip, new Rectangle(canvasWidth+16,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
	}
	private void entry2Shooter() {
		int clip = anim.startClip("simpleClip16", new Rectangle(240,0,16,16), 1.0);
		//anim.queueClipChange(clip, new Rectangle(0,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
		anim.queueClipChange(clip, new Rectangle(240/2,0,shootEndWidth/2+16,16), null, 1.0, 1400, 0, false);
		anim.queueClipChange(clip, new Rectangle(0,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
	}
	private void exit2Shooter() {
		int clip = anim.startClip("simpleClip16", new Rectangle(0,0,16,16), 1.0);
		//anim.queueClipChange(clip, new Rectangle(240,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
		anim.queueClipChange(clip, new Rectangle(240/2,0,shootEndWidth/2+16,16), null, 1.0, 1400, 0, false);
		anim.queueClipChange(clip, new Rectangle(240,0,shootEndWidth,16), null, 0.0, 1400, 0, true);
	}


	private void bigFill() {
		//soundController.playSingleBay("test_1.wav", 600.0, 1.0f); // plays a sound out of the speaker nearest to the x value provided
		// create clip off stage left
		int clip = anim.startClip("simpleClip16", new Rectangle(-14,0,16,16), 1.0);
		// expand clip1 to full screen
		anim.queueClipChange(clip, new Rectangle(0,0,(int)canvasWidth,(int)canvasHeight), null, null, 3000, 0, false);
		// retract clip to right
		anim.queueClipChange(clip, new Rectangle(canvasWidth,0,16,16), null, null, 3000, 1000, true);
	}

	private void eggExpand(double x) {
		int startWidth = 2;
		int endWidth = 64;
		int clip = anim.startClip("simpleClip16", new Rectangle((int)x - startWidth/2,0,startWidth,16), 1.0);

		anim.queueClipChange(clip, new Rectangle((int)x - endWidth/2,0,endWidth,16), null, null, 1200, 0, false);
		anim.queueClipChange(clip, new Rectangle((int)x - startWidth/2,0,startWidth,16), null, null, 800, 0, true);
		//logger.info("Created Egg Expand at x = " + x);
	}

	private void eggWave(double x){
		int clip = anim.startClip("imageClipWave", new Rectangle((int)x-16,0,48,16), 0.0);
		//clip mask not working?
		//anim.queueClipChange(clip, new Rectangle((int)x - 64/2,0,64,16), new Rectangle((int)x - 8,0,16,16), 1.0, 5, 0, false);
		anim.queueClipChange(clip, null, null, 1.0, 800, 0, false);
		anim.queueClipChange(clip, new Rectangle((int)x - 32,0,32,16), null, 1.0, 1600, 0, false);
		anim.queueClipChange(clip, null, null, 0.0, 300, 0, true);
		//logger.info("Created Egg Wave at x = " + x);
	}

	private void eggSparkle(double x){
		int clip = anim.startClip("sparkleClip32", new Rectangle((int)x-16,0,32,16), 0.0);
		//anim.queueClipChange(clip, null, null, 1.0, 1700, 0, false); //make it smaller
		anim.queueClipChange(clip, null, null, 1.0, 1000, 0, false); //fadein
		anim.queueClipChange(clip, null, null, 0.0, 1000, 1200, true); //fadeout
		//logger.info("Created Egg Sparkle at x = " + x);
	}

	private void Tracer(double x) {
		int blockWidth = 10;
		int clip = anim.startClip("testClip", new Rectangle((int)x - blockWidth/2,0,blockWidth,16), 1.0);

		anim.queueClipChange(clip, null, null, 0.0, 4000, 1000, true);

	}

     */




    /************************* Main Loop ******************************/

    public void run() {
        timer.start();
        curTime = System.currentTimeMillis();

        while (isRunning) {

            model.poll();

            // ELU
            if (updateLighting){
                try {
                    canvas.sync(AnimationManager.toPixels(anim.getStage(), anim.getStageDimensions().width, anim.getStageDimensions().height));
                    elu.syncAllLights();
                } catch (InvalidPixelGrabException e) {
                    // TODO Auto-generated catch block
                    e.printStackTrace();
                }
            }

            // Update the GUI Panel
            ef.update();

            //Thread ops
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