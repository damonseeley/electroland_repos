package net.electroland.edmonton.core;

/**
 * @title	"EIA" by Electroland
 * @author	Damon Seeley & Bradley Geilfuss
 */

import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;
import java.io.IOException;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Hashtable;
import java.util.List;

import net.electroland.ea.AnimationManager;
import net.electroland.edmonton.core.model.OneEventPerPeriodModelWatcher;
import net.electroland.edmonton.core.ui.EIAFrame;
import net.electroland.eio.IOManager;
import net.electroland.eio.IState;
import net.electroland.eio.model.Model;
import net.electroland.eio.model.ModelEvent;
import net.electroland.eio.model.ModelListener;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.InvalidPixelGrabException;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAMainConductor2 extends Thread implements ActionListener, ModelListener {

    static Logger logger = Logger.getLogger(EIAMainConductor2.class);

    private ElectrolandProperties props, propsStatic;
    private ELUManager elu;
    private ELUCanvas2D canvas;
    private EIAClipPlayer2 clipPlayer2;
    private IOManager eio;
    private SoundController soundController;
    private AnimationManager anim;
    private EIAFrame ef;
    private Model model;
    private TrafficFlowAnalyzer tfa;

    private boolean updateLighting = true;
    public int canvasHeight, canvasWidth;
    public Hashtable<String, Object> context;

    //Thread stuff
    public static boolean isRunning;
    private static float framerate;
    private static FrameTimer timer;

    public EIAMainConductor2()
    {
        context = new Hashtable<String, Object>();

        String propsFileName = "EIA.properties";
        logger.info("EIAMain loading " + propsFileName);
        props = new ElectrolandProperties(propsFileName);
        context.put("props",props);

        String propsStaticFileName = "EIA-static-elements.properties";
        logger.info("EIAMain loading " + propsStaticFileName);
        propsStatic = new ElectrolandProperties(propsStaticFileName);
        context.put("propsStatic",propsStatic);

        elu = new ELUManager();
        eio = new IOManager();

        boolean eioplayback = false;
        try {
            eioplayback = Boolean.parseBoolean(props.getOptional("settings", "eiomode", "playback"));
        } catch (OptionException e) {
            eioplayback = false;
            e.printStackTrace();
        }

        try {
            elu.load("EIA-ELU-3ch.properties");
            
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

        clipPlayer2 = new EIAClipPlayer2(context);


        /******** Model, Watchers & Timers ********/
        model = new Model();
        model.addModelListener(this);
        
        // watchers per istate
        ElectrolandProperties clipNames = new ElectrolandProperties("EIA-clipSchedule.properties");
        for (IState state : eio.getIStates())
        {
        	String clip = clipNames.getRequired("sensor", state.getID(), "clipName");
        	int clipTiming = clipNames.getRequiredInt("sensor", state.getID(), "clipTiming");
        	model.addModelWatcher(new OneEventPerPeriodModelWatcher(clip, clipTiming), "showwatcher" + state.getID(), state);
        }
        
        tfa = new TrafficFlowAnalyzer(2,30000,300000);
        context.put("tfa", tfa);
        
        /******** GUI ********/
        ef = new EIAFrame(Integer.parseInt(props.getRequired("settings", "global", "guiwidth")),Integer.parseInt(props.getRequired("settings", "global", "guiheight")),context);
        ef.addButtonListener(this);

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

        if (evt.getSource() instanceof OneEventPerPeriodModelWatcher){

        	OneEventPerPeriodModelWatcher src = (OneEventPerPeriodModelWatcher)evt.getSource();
        	
        	// update Traffic Flow Analyer
        	// moved below
        	//tfa.trip(src.getStates());

            // play clip
            Method[] allMethods = clipPlayer2.getClass().getDeclaredMethods();
            for (Method m : allMethods) {
                if (m.getName().equals(src.getClipName()))
                {
                    double xloc = src.getStates().iterator().next().getLocation().x;
                    try {
                        m.invoke(clipPlayer2, xloc);
                        tfa.trip(xloc);
                    } catch (IllegalArgumentException e) {
                        e.printStackTrace();
                    } catch (IllegalAccessException e) {
                        e.printStackTrace();
                    } catch (InvocationTargetException e) {
                        e.printStackTrace();
                    }
                }
            }
        }
        
        ef.updateFlowLabels(tfa.getPM1Flow(), tfa.getPM2Flow(), tfa.getPM1Avg(), tfa.getPM2Avg());

    } 

    /************************* Test Event Handlers ******************************/
    public void actionPerformed(ActionEvent e) {
       
        if ("showHideGfx".equals(e.getActionCommand())) {
        	ef.showHideGfx();
        	logger.info(e.getActionCommand());
        } else if ("testShow".equals(e.getActionCommand())){
        	clipPlayer2.testClip(Math.random()*625.0);
        } else if ("pm1".equals(e.getActionCommand())){
            //logger.info("People Mover 1 total trips for 30s = " +tfa.getPM1Flow());
            tfa.logpm1();
        } else if ("pm2avg".equals(e.getActionCommand())){
            logger.info("People Mover 2 total trips for 30s = " +tfa.getPM2Flow());
        }
        
        
    }

    /************************* Main Loop ******************************/
    public void run() {

    	timer.start();

        while (isRunning) {

        	// poll the sensors
            model.poll();

            // update the lighting system
            if (updateLighting){
                try {
                    canvas.sync(AnimationManager.toPixels(anim.getStage(), anim.getStageDimensions().width, anim.getStageDimensions().height));
                    elu.syncAllLights();
                } catch (InvalidPixelGrabException e) {
                    e.printStackTrace();
                }
            }

            // Update the GUI Panel
            ef.update();

            //Thread ops
            timer.block();
        }
    }

    public static void main(String args[])
    {
        new EIAMainConductor2();
    }
}