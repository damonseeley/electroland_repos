package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.awt.Color;
import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.util.Hashtable;
import java.util.Timer;
import java.util.TimerTask;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.edmonton.core.model.LightBlip;
import net.electroland.edmonton.core.model.LastTrippedModelWatcher;
import net.electroland.edmonton.core.model.LightBlipModelWatcher;
import net.electroland.edmonton.core.model.ScreenSaverModelWatcher;
import net.electroland.edmonton.core.model.TrackerBasicModelWatcher;
import net.electroland.edmonton.core.sequencing.SimpleSequencer;
import net.electroland.edmonton.core.ui.EIAFrame;
import net.electroland.eio.IOManager;
import net.electroland.eio.model.Model;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor2 extends Thread implements ActionListener, ModelListener {

    static Logger logger = Logger.getLogger(EIAMainConductor2.class);

    private int fadeDuration = 2000;
    private int inactivityThreshold = 1000 * 60;
    private ElectrolandProperties props;
    private ELUManager elu;
    private boolean updateLighting = true;
    private boolean track;
    private boolean screensaver = true;
    private boolean kickstart = false;
    private int kickdelay = 10000;
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
    //private ModelWatcher stateToBright,entry1,exit1,entry2,exit2,egg1,egg2,egg3,egg4;
    private TrackerBasicModelWatcher tracker;
    private LastTrippedModelWatcher tripRecord;
    private ScreenSaverModelWatcher screenSaver;
    private LightBlipModelWatcher blipper;

    private int stateToBrightnessClip;

    private EIAGenSoundPlayer gsp;

    //Thread stuff
    public static boolean isRunning;
    private static float framerate;
    private static FrameTimer timer;
    public static long curTime = System.currentTimeMillis(); //  time of frame start to aviod lots of calls to System.getcurentTime()
    public static long elapsedTime = -1; //  time between start of cur frame and last frame to avoid re calculating passage of time allover the place

    private boolean isLive = false;

    public EIAMainConductor2()
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
        
        try {
            kickstart = Boolean.parseBoolean(props.getOptional("settings", "sequencing", "kickstart"));
            kickdelay = props.getOptionalInt("settings", "sequencing", "kickdelay");
            isLive = true;
        } catch (OptionException e) {
            // TODO Auto-generated catch block
            kickstart = false;
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

        String seqpropsfile = "EIA-sequencer.properties";
        sequencer = new SimpleSequencer(seqpropsfile, context);
        context.put("sequencer", sequencer);
        context.put("seqpropsfile", seqpropsfile);

        soundController = new SoundController(context);
        context.put("soundController", soundController);
        // for generative show test
        gsp = new EIAGenSoundPlayer(soundController);


        clipPlayer = new EIAClipPlayer(anim,elu,soundController);
        context.put("clipPlayer", clipPlayer);



        /******** Model, Watchers & Timers ********/
        model = new Model();
        model.addModelListener(this);
        model.addModelListener(sequencer);

        if (track) {
            tracker = new TrackerBasicModelWatcher(context); //starting with vals of 64 which is what was used in TestConductor (single value)
            model.addModelWatcher(tracker, "tracker", eio.getIStates());
            context.put("tracker", tracker);
        }

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
    
    
    
    /************************* Model Handlers ******************************/

    @Override
    public void modelChanged(ModelEvent evt) {

        if (screensaver && evt.getSource() instanceof ScreenSaverModelWatcher){
            logger.info("got screen saver event at " + System.currentTimeMillis());
            if (((ScreenSaverModelWatcher)evt.getSource()).isQuiet())
            {
                this.goQuiet();
            }else{
                this.goLive();
            }
        }
        
        
        
        


    } 






    /************************* Test Event Handlers ******************************/

    // handle the actions from test buttons
    public void actionPerformed(ActionEvent e) {
        logger.info(e.getActionCommand());

        if ("startShow1".equals(e.getActionCommand())) {
            // to change this behavior, change goLive().
            this.goLive();
        }
        if ("startShow2".equals(e.getActionCommand())) {
            // to change this behavior, change goQuiet().
            this.goQuiet();
        }
        if ("testShow".equals(e.getActionCommand())) {
            sequencer.play("testShow");
            //this.goLive();
        }
        if ("stopSeq".equals(e.getActionCommand())) {
            sequencer.stop();
            isLive = false;
            clipPlayer.live.deleteChildren();
            clipPlayer.quiet.deleteChildren();
            soundController.fadeAll(500);
        }

    }

    public void goQuiet()
    {
        // TECHNICALLY: THIS IS ALL YOU NEED.
        //sequencer.play(sequencer.quietShowId);
        // then comment block below out.
        if (isLive){
            logger.info("go Screensaver");
            isLive = false;
            soundController.fadeAll(500);
            clipPlayer.live.fadeOut(500).deleteChildren();
            clipPlayer.quiet.fadeIn(0);
            sequencer.play(sequencer.quietShowId);
        }else{
            logger.warn("attempt to start screensaver while already screensaving (declined).");
        }
    }

    public void goLive(){
        // TECHNICALLY: THIS IS ALL YOU NEED.
        //sequencer.play(sequencer.liveShowId);
        // then comment block below out.
        if (!isLive){
            logger.info("go Live");
            isLive = true;
            soundController.fadeAll(500);
            clipPlayer.quiet.fadeOut(500).deleteChildren();
            clipPlayer.live.fadeIn(0);
            sequencer.play(sequencer.liveShowId);
        }else{
            logger.warn("attempt to go live while already live (declined).");
        }
    }




    /************************* Local Animations ******************************/


   



    /************************* Main Loop ******************************/

    public void run() {
        timer.start();
        curTime = System.currentTimeMillis();

        if (screensaver && kickstart)
        {
            try {
                // wait until we're sure everything has started so the audio
                // can start cleanly
                Thread.sleep(kickdelay);
                // start screensaver
                logger.info("kickstarting screensaver.");
                this.goQuiet();
            } catch (InterruptedException e) {
                e.printStackTrace();
            }
        }else{
            logger.info("no kickstart.");
        }
        
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
        new EIAMainConductor2();
    }




}