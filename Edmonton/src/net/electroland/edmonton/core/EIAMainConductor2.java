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
import java.util.Hashtable;

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

    private ElectrolandProperties props, propsGlobal;
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

        String propsFileName = "EIA-local.properties";
        logger.info("EIAMain loading " + propsFileName);
        props = new ElectrolandProperties(propsFileName);
        context.put("props",props);

        String propsGlobalFileName = "EIA-global.properties";
        logger.info("EIAMain loading " + propsGlobalFileName);
        propsGlobal = new ElectrolandProperties(propsGlobalFileName);
        context.put("propsGlobal",propsGlobal);

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
        
        tfa = new TrafficFlowAnalyzer(5,10000,120000);
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

            double xloc = src.getStates().iterator().next().getLocation().x;
            playClip(src.getClipName(), xloc);
        }

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
        } else if ("random".equals(e.getActionCommand())){
            double bottom = props.getRequiredDouble("settings", "random", "bottom");
            double top    = props.getRequiredDouble("settings", "random", "top");
            double range  = top - bottom;
            double random = Math.random() * range + bottom;

            playClip(ef.getSelectedClip(), random);
        }
    }

    public void playClip(String name, double loc){

        try {

            logger.info("Running clipPlayer2." + name + '(' + loc + ')');
            Method m = clipPlayer2.getClass().getMethod(name, double.class);
            m.invoke(clipPlayer2, loc);

        } catch (SecurityException e1) {
            e1.printStackTrace();
        } catch (NoSuchMethodException e1) {
            e1.printStackTrace();
        } catch (IllegalArgumentException e1) {
            e1.printStackTrace();
        } catch (IllegalAccessException e1) {
            e1.printStackTrace();
        } catch (InvocationTargetException e1) {
            e1.printStackTrace();
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
            ef.updateFlowLabels(tfa.getCurAvgTime(), tfa.getRunAvgTime(), tfa.getPM1Flow(), tfa.getPM2Flow(), tfa.getPM1Avg(), tfa.getPM2Avg());
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