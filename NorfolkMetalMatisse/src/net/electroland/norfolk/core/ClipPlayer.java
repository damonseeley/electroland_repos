package net.electroland.norfolk.core;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;
import java.util.Timer;
import java.util.TimerTask;

import javax.vecmath.Point2d;

import net.electroland.ea.Animation;
import net.electroland.ea.AnimationListener;
import net.electroland.ea.Clip;
import net.electroland.ea.Sequence;
import net.electroland.eio.InputChannel;
import net.electroland.norfolk.sound.SimpleSoundManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;

import org.apache.log4j.Logger;

public class ClipPlayer implements AnimationListener {

    private static Logger logger = Logger.getLogger(ClipPlayer.class);
    private Animation eam;
    private SimpleSoundManager ssm;
    private ELUManager elu;
    private Map<String, Fixture> channelToFixture;
    private Collection<Method> globalClips;

    private Timer chordTimer;
    int chordIndex;
    int chordIndexMax;
    long chordDur;

    private enum Message {SCREENSAVER, IVASE_THROB, SSVASE_THROB, COBRA_THROB, LEAVES, SPARKLE}

    private int detOffset = 2;

    public ClipPlayer(Animation eam, SimpleSoundManager ssm, ELUManager elu, ElectrolandProperties props){

        this.eam = eam;
        this.eam.addListener(this);
        this.ssm = ssm;
        this.elu = elu;
        this.ssm.load(props);
        this.configure(props);

        chordIndex = 1;
        chordIndexMax = 3;
        chordDur = 5000; //4 seconds for each chord
        chordTimer = new Timer();
        chordTimer.schedule(new chordTimerTask(), chordDur, chordDur);

        initMasterClips();
        initScreensaver();
        initInteractive();
    }


    class chordTimerTask extends TimerTask{

        public void run(){
            if (chordIndex < chordIndexMax ) {
                chordIndex++;
            } else {
                chordIndex = 1;
            }
            //logger.info("Changed chordIndex to " + chordIndex);
        }
    }

    public void configure(ElectrolandProperties props){

        channelToFixture = new HashMap<String, Fixture>();

        for (ParameterMap mappings : props.getObjects("channelFixture").values()){
            String channelId = mappings.get("channel");
            String fixtureId = mappings.get("fixture");
            Fixture fixture = this.getFixture(fixtureId); // can be null
            channelToFixture.put(channelId, fixture);
        }

        globalClips = getGlobalClips(true);
    }



    /**
     * The boolean is irrelevent. Just trying to get it not to show up on the
     * list of methods with no args.
     * @param foo
     * @return
     */
    public Collection<Method> getGlobalClips(boolean foo){
        Method[] methods = this.getClass().getDeclaredMethods();
        ArrayList<Method> globalClips = new ArrayList<Method>();
        for (Method method:methods)
        {
            if (method.getParameterTypes().length == 0){
                globalClips.add(method);
            }
        }
        return globalClips;
    }

    public void play(String globalClipName){
        for (Method method  : globalClips){
            if (method.getName().equals(globalClipName)){
                try {
                    logger.debug("play '" + globalClipName + "'");
                    method.invoke(this);
                } catch (IllegalArgumentException e) {
                    logger.warn(e);
                } catch (IllegalAccessException e) {
                    logger.warn(e);
                } catch (InvocationTargetException e) {
                    logger.warn(e);
                }
            }
        }
    }

    public void play(String clipName, InputChannel channel){

        Fixture f = channelToFixture.get(channel.getId());

        try {

            logger.debug("play '" + clipName + "' for channel " + channel);
            Method method = this.getClass().getMethod(clipName, Fixture.class);
            if (f == null){
                method.invoke(this);
            }else{
                method.invoke(this, f);
            }
        } catch (IllegalArgumentException e) {
            logger.warn(e);
        } catch (IllegalAccessException e) {
            logger.warn(e);
        } catch (InvocationTargetException e) {
            logger.warn(e);
        } catch (NoSuchMethodException e){
            logger.warn(e);
        }
    }


    /** CLIP SETUP ****************************/

    private void initMasterClips(){
        //add interactive first so screensaver covers it up
        interactive = eam.addClip(null, null, 0, 0, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 1.0f);
        screensaver = eam.addClip(null, null, 0, 0, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 1.0f);
    }

    /** SCREENSAVER LOGIC ****************************/

    public void enterScreensaverMode(int millis){
        logger.debug("enter screensaver");
        screensaver.fadeIn(millis);
    }

    public void exitScreensaverMode(int millis){
        logger.debug("exit screensaver.");
        screensaver.fadeOut(millis);
    }

    /**
     * When screensavers complete, this gets called. This is the chance
     * to decide what screensaver to play next.
     */
    @Override
    public void messageReceived(Object message) {

        if (message instanceof Message){

            switch((Message)message){
            case SCREENSAVER:
                ssMultiClouds();
                break;
            case SSVASE_THROB:
                ssVaseThrob();
                break;
            case COBRA_THROB:
                ssCobraThrob();
                break;
            case LEAVES:
                ssGreenLeaves();
                break;
            case SPARKLE:
                ssSparkle();
                break;
            case IVASE_THROB:
                iVaseThrob();
                break;
            }
        }
    }


    /** SCREENSAVER CLIPS ****************************/

    //nested clips
    private Clip screensaver; //master clip for all screensaver animation
    private Clip ssVase;
    private Clip ssFlora;
    private Clip ssCobras;
    private Clip ssLeaves;

    // locations of stuff
    private int vaseVMin = 0;
    private int vaseVMax = 17;
    private int elementsVMax = 175;
    private int cobrasVMin = 176;
    private int cobrasVMax = 200;
    private int leavesX = 130; //not right
    private int leavesY = 28;
    private int leavesWidth = 45;
    private int leavesHeight = 20;

    // alpha min max
    private float ssVaseThrobMax = 0.7f;
    private float ssVaseThrobMin = 0.15f;

    //overall throb timing for ss elements
    private int throbPeriod = 6000;
    private int holdPeriod = 700;


    private void initScreensaver() {

        ssVase = screensaver.addClip(null, null, 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, 1.0f);
        ssFlora = screensaver.addClip(null, null, 0, vaseVMax, eam.getFrameDimensions().width, elementsVMax-vaseVMax, 1.0f);
        ssCobras = screensaver.addClip(null, null, 0, elementsVMax, eam.getFrameDimensions().width, cobrasVMax-elementsVMax, 1.0f);
        ssLeaves = screensaver.addClip(null, null, leavesX, leavesY, leavesWidth, leavesHeight, 1.0f);

        //TESTS for regions
        //ssVase.addClip(null, Color.getHSBColor(.0f, 1.0f, 1.0f), 0, 0, eam.getFrameDimensions().width, vaseVMax, 1.0f);
        //ssFlora.addClip(null, Color.getHSBColor(.3f, 1.0f, 1.0f), 0, 0, eam.getFrameDimensions().width, elementsVMax, 1.0f);
        //ssCobras.addClip(null, Color.getHSBColor(.6f, 1.0f, 1.0f), 0, 0, eam.getFrameDimensions().width, cobrasVMax, 1.0f);
        //ssLeaves.addClip(null, Color.getHSBColor(.33f, 1.0f, 1.0f), 0, 0, leavesWidth, leavesHeight, 0.5f);

        //init
        ssInitCobras();

        //start the constant clips
        ssVaseThrob();
        ssCobraThrob();
        ssGreenLeaves();
        //screensaverMultiClouds(); //OR
        ssSparkle();
    }

    public void ssSparkle(){

        int delay = 0;
        ArrayList<Clip> clips = new ArrayList<Clip>();
        Clip last = null;

        for (Fixture f : elu.getFixtures()){
            if (f.getName().toLowerCase().startsWith("f") || f.getName().toLowerCase().startsWith("b")){
                last = sparklet(f, delay += this.throbPeriod);
                clips.add(last);
            }
        }
        for (Clip c : clips){
            if (c == last){
                c.announce(Message.SPARKLE).deleteWhenDone();
            }else{
                c.deleteWhenDone();
            }
        }
    }

    public Clip sparklet(Fixture fixture, int pause){

        logger.info("start sparkle on " + fixture.getName() + " with pause of " + pause);
        /* ranges
         * 0-0.2
         * .77-1.0
         */
        float hueMin = 0.7f;
        float hueDelta = 0.5f;
        float randHue = hueMin + (float)((Math.random() * hueDelta));
        Clip f = eam.addClip(null, Color.getHSBColor(randHue, .99f, .99f),(int)fixture.getLocation().x - 4,(int)fixture.getLocation().y - 4, 10, 10, 0.0f);

        Sequence huechange = new Sequence();
        float hueChangeRange = 0.25f;
        float huernd = hueChangeRange - (float)(Math.random() * hueChangeRange * 2);
        huechange.hueBy(huernd);
        huechange.duration(throbPeriod);
        huechange.alphaTo(0.0f);

        f.pause(pause).fadeIn(throbPeriod).pause(holdPeriod).queue(huechange);

        return f;
    }

    private void ssGreenLeaves() {
        int duration   = 30000;
        int width     = 600;
        
        Clip black = ssLeaves.addClip(null, Color.getHSBColor(.0f, .0f, .0f), 0, 0, leavesWidth, leavesHeight, 1.0f);

        Clip leafPulse    = ssLeaves.addClip(null, 
                null, 
                0, 0, 
                leavesWidth, leavesHeight, 
                1.0f);
        

        Clip leafGreen    = leafPulse.addClip(eam.getContent("gradient_600_greenyellow2"), 
                null, 
                -width, 0, 
                width, leavesHeight, 
                1.0f);

        Sequence sweep = new Sequence();
        //sweep.yTo(eam.getFrameDimensions().height).duration(duration);
        sweep.xTo(leavesWidth).duration(duration);
        
        leafGreen.queue(sweep).announce(Message.LEAVES).fadeOut(2000).deleteWhenDone();

        Sequence slowPulseOut = new Sequence();
        slowPulseOut.alphaTo(0.2f).duration(throbPeriod);

        Sequence slowPulseIn = new Sequence();
        slowPulseIn.alphaTo(0.99f).duration(throbPeriod);

        leafPulse.queue(slowPulseOut).pause(500).queue(slowPulseIn).queue(slowPulseOut).pause(500).queue(slowPulseIn).queue(slowPulseOut).queue(slowPulseIn).pause(500).deleteWhenDone();  

        
    }


    public void ssMultiClouds(){

        int duration   = 60000;
        int height     = 800;

        Clip clouds    = ssFlora.addClip(eam.getContent("clouds_200x800_multi_angle"), 
                null, 
                0, -height, 
                eam.getFrameDimensions().width, height, 
                0.0f);

        int fadeInTime = 4000;
        Sequence sweep = new Sequence();
        //sweep.yTo(eam.getFrameDimensions().height).duration(duration);
        sweep.yTo(-height + (height * fadeInTime/duration));
        sweep.alphaTo(1.0f).duration(fadeInTime).newState();
        sweep.yTo(0).duration(duration-fadeInTime);

        clouds.queue(sweep).announce(Message.SCREENSAVER).fadeOut(fadeInTime*2).deleteWhenDone();
    }


    public void ssVaseThrob() {
        //logger.info("VASE THROB STARTED");
        Clip black = ssVase.addClip(null, Color.getHSBColor(.0f, .0f, .0f), 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, 1.0f);
        Clip vaseBlue = ssVase.addClip(null, Color.getHSBColor(.55f, .99f, .99f), 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, ssVaseThrobMin);

        Sequence slowPulseOut = new Sequence();
        slowPulseOut.hueBy(0.05f).duration(throbPeriod/2);
        slowPulseOut.alphaTo(ssVaseThrobMin).duration(throbPeriod/2);

        Sequence slowPulseIn = new Sequence();
        slowPulseIn.hueBy(-0.05f).duration(throbPeriod/2);
        slowPulseIn.alphaTo(ssVaseThrobMax).duration(throbPeriod/2);

        vaseBlue.queue(slowPulseIn).queue(slowPulseOut).announce(Message.SSVASE_THROB).pause(500).deleteWhenDone();    
    }


    private int cobraIndex = 0;
    HashMap<Integer,Point2d> cobraLocs = new HashMap<Integer, Point2d>();

    public void ssInitCobras() {

        ArrayList<Fixture> fixs = new ArrayList<Fixture>();
        for (Fixture f : elu.getFixtures()){
            fixs.add(f);
        }

        Point2d c1 = new Point2d();
        Point2d c2 = new Point2d();
        Point2d c3 = new Point2d();

        for (Fixture f : elu.getFixtures()){
            if (f.getName().equals("c01a")) {
                c1.x = f.getLocation().x;
                c1.y = f.getLocation().y;
                //logger.info("c1 " + c1);
            } else if (f.getName().equals("c02a")) {
                c2.x = f.getLocation().x;
                c2.y = f.getLocation().y;
                //logger.info("c2 " + c2);
            } else if (f.getName().equals("c03a")) {
                c3.x = f.getLocation().x;
                c3.y = f.getLocation().y;
                //logger.info("c3 " + c3);
            }
        }

        cobraLocs.put(0, c1);
        cobraLocs.put(1, c2);
        cobraLocs.put(2, c3);
    }

    public void ssCobraThrob() {
        //logger.info("COBRATHROB at x: " + cobraLocs.get(cobraIndex).x);

        int margin = 2;
        int width = 12;
        int height = 24;
        Clip cobraBlue = ssCobras.addClip(null, Color.getHSBColor(.6f, .99f, .99f),
                (int)(cobraLocs.get(cobraIndex).x) - margin,
                (int)(cobraLocs.get(cobraIndex).y) - margin - cobrasVMin,
                width,
                height,
                0.0f);

        Sequence slowPulseIn = new Sequence();
        slowPulseIn.hueBy(-0.05f).duration(throbPeriod);
        slowPulseIn.alphaTo(1.0f).duration(throbPeriod);

        Sequence slowPulseOut = new Sequence();
        slowPulseOut.hueBy(0.05f).duration(throbPeriod);
        slowPulseOut.alphaTo(0.0f).duration(throbPeriod);

        cobraBlue.queue(slowPulseIn).pause(holdPeriod).queue(slowPulseOut).announce(Message.COBRA_THROB).deleteWhenDone();

        if (cobraIndex < 2) {
            cobraIndex++;
        } else {
            cobraIndex = 0;
        }
    }






    /** BIG SHOWS AND COMBO CUES ****************************/

    /** Accents ****************************/

    private float sensorPulseMax= 0.8f;
    private float sensorPulseMin= 0.0f;

    public void iPulseVaseSensor() {
        //logger.info("IVASE PULSE");
        Clip vasePulse = eam.addClip(null, Color.getHSBColor(.55f, .99f, .99f), 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, sensorPulseMin);

        int dur = 150;
        Sequence pulseIn = new Sequence();
        pulseIn.alphaTo(sensorPulseMax).duration(dur);
        Sequence pulseOut = new Sequence();
        pulseOut.alphaTo(sensorPulseMin).duration(dur*2);

        vasePulse.queue(pulseIn).queue(pulseOut).fadeOut(300).deleteWhenDone();    
    }

    public void freakOut() {

        ssm.playSound("freakout");

        int stripWidth    = 20;
        int enterDuration = 1000;
        int danceDuration = 6000;
        int width = eam.getFrameDimensions().width;
        int height = eam.getFrameDimensions().height;

        // enter
        Clip bgbk = eam.addClip(Color.BLACK, 0, 0, width, height, 0.0f);
        Clip bgor = eam.addClip(Color.ORANGE, 0, 0, width, height, 0.0f);

        Sequence lightUp = new Sequence();
        lightUp.alphaTo(1.0f).duration(enterDuration);

        bgbk.queue(lightUp).pause(10000).fadeOut(500).deleteWhenDone();
        bgor.queue(lightUp).pause(danceDuration).fadeOut(500).deleteWhenDone();

        // dance
        Clip stage = eam.addClip(-width, 0, width * 2, height, 1.0f);
        ArrayList<Clip>oddClips = new ArrayList<Clip>();
        ArrayList<Clip>evenClips = new ArrayList<Clip>();

        for (int i = 0; i < ((width * 2) / stripWidth); i ++){
            if (i % 2 == 0){
                evenClips.add(stage.addClip(Color.RED, i * stripWidth, 0, stripWidth, height, 1.0f));
            }else{
                oddClips.add(stage.addClip(Color.RED, i * stripWidth, 0, stripWidth, height, 0.0f));
            }
        }

        // move the whole thing
        Sequence shiftRight = new Sequence();
        shiftRight.xTo(0).duration(4 * danceDuration);
        stage.pause(enterDuration).queue(shiftRight).fadeOut(500).deleteWhenDone();

        Sequence flash = new Sequence();
        flash.alphaTo(1.0f).duration(150).newState();
        flash.alphaTo(0.0f).duration(150).newState();
        flash.alphaTo(1.0f).duration(150).newState();
        flash.alphaTo(0.0f).duration(250).newState();
        

        // flash evens
        for (Clip c : evenClips){
            c.queue(new Sequence().pause(enterDuration + 200));
            c.queue(flash).pause(400).queue(flash).queue(new Sequence().hueBy((float)Math.random()));
            c.queue(new Sequence().pause(500));
            c.queue(flash).pause(400).queue(flash).queue(new Sequence().hueBy((float)Math.random()));
            c.deleteWhenDone();
        }

        // flash odds
        for (Clip c : oddClips){
            c.queue(new Sequence().pause(enterDuration));
            c.queue(flash).pause(400).queue(flash).queue(new Sequence().hueBy((float)Math.random()));
            c.queue(new Sequence().pause(500));
            c.queue(flash).pause(400).queue(flash).queue(new Sequence().hueBy((float)Math.random()));
            c.queue(new Sequence().pause(500));
            c.queue(flash).pause(400).queue(flash).queue(new Sequence().hueBy((float)Math.random()));
            c.deleteWhenDone();
        }

        // some brighter lights
        Clip brights = eam.addClip(Color.WHITE, 0, 0, width, height, 0.0f);
        
        brights.pause(2000).queue(flash).pause(400).queue(flash);
        brights.pause(2000).queue(flash).pause(400).queue(flash);
        brights.pause(750).queue(flash);
        brights.deleteWhenDone();
    }

    /** Vase ****************************/

    //nested clips
    private Clip interactive; 
    private Clip iVase;
    private Clip iFlora;

    public void initInteractive(){
        //add iVase
        iVase = interactive.addClip(null, null, 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, 1.0f);
        iFlora = interactive.addClip(null, null, 0, vaseVMax, eam.getFrameDimensions().width, elementsVMax, 1.0f);

        //call vase throb
        iVaseThrob();

    }

    public void iVaseThrob() {
        //logger.info("iVASE THROB STARTED");

        float iVaseMin = 0.4f;
        float iVaseMax = 0.6f;
        Clip black = iVase.addClip(null, Color.getHSBColor(.0f, .0f, .0f), 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, 1.0f);
        Clip vaseBlue = iVase.addClip(null, Color.getHSBColor(.55f, .99f, .99f), 0, vaseVMin, eam.getFrameDimensions().width, vaseVMax, iVaseMin);

        Sequence slowPulseOut = new Sequence();
        slowPulseOut.hueBy(0.05f).duration(throbPeriod);
        slowPulseOut.alphaTo(iVaseMin).duration(throbPeriod);

        Sequence slowPulseIn = new Sequence();
        slowPulseIn.hueBy(-0.05f).duration(throbPeriod);
        slowPulseIn.alphaTo(iVaseMax).duration(throbPeriod);

        vaseBlue.queue(slowPulseIn).queue(slowPulseOut).announce(Message.IVASE_THROB).pause(500).deleteWhenDone();    
    }


    /** Bigger shows ****************************/


    public void comboCobrasOrange(){
        //ssm.playGroupRandom("7");

        int duration = 3000;
        int width = 600;
        int vLoc = 174; // start of cobras
        int vHeight = 28; // cover all cobras

        Clip parent = eam.addClip(null, null, -width*2, vLoc, width*2, vHeight, 1.0f);
        parent.addClip(eam.getContent("bar1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), 0, 0, width, vHeight, 1.0f);
        parent.addClip(eam.getContent("bar1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), width, 0, width, vHeight, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width).duration(duration);
        sweep.xTo(0).duration(duration);

        parent.queue(sweep).fadeOut(500).deleteWhenDone(); 
    }


    public void comboLeavesGreen(){
        //ssm.playGroupRandom("7");

        int duration = 5000;
        int width = 600;
        int vLoc = 30; // start of cobras
        int vHeight = 12; // cover all cobras

        Clip parent = eam.addClip(null, null, -width + 120, vLoc, width, vHeight, 1.0f);
        parent.addClip(eam.getContent("gradient_600_greenyellow"), null, 0, 0, width, vHeight, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width).duration(duration);
        //sweep.xTo(0).duration(duration);

        parent.queue(sweep).fadeOut(500).deleteWhenDone(); 

    }

    public void comboTulipsBlueCyan(){
        //ssm.playGroupRandom("7");

        int duration = 8000;
        int width = 600;
        int vLoc = 146; // start of cobras
        int vHeight = 12; // cover all cobras

        Clip parent = eam.addClip(null, null, -width*2, vLoc, width*2, vHeight, 1.0f);
        parent.addClip(eam.getContent("gradient_600_bluecyan"), null, 0, 0, width, vHeight, 1.0f);
        parent.addClip(eam.getContent("gradient_600_bluecyan"), null, width, 0, width, vHeight, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width).duration(duration);
        //sweep.xTo(0).duration(duration);

        parent.queue(sweep).fadeOut(500).deleteWhenDone(); 

    }
    
    public void allElementMulti() {
        //ssm.playGroupRandom("8");
        
        Clip c1 = eam.addClip(eam.getContent("allelements_multi"), Color.getHSBColor(.0f, .99f, .0f), 0, 0, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 1.0f);

        Sequence fade = new Sequence();
        fade.brightnessTo(0.0f).duration(2000);

        c1.pause(2000).queue(fade).fadeOut(1500).deleteWhenDone();
    }


    public void vertWavesRedMag(){
        ssm.playGroupRandom("8");
        //ssm.playGroupRandom("6");

        int duration = 5000;

        int height = 600;
        //Clip c = eam.addClip(Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c1 = eam.addClip(eam.getContent("grad1200_vert_three_red_mag"), Color.getHSBColor(.4f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("grad1200_vert_three_red_mag"), Color.getHSBColor(.5f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);        //Clip c4 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.7f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(duration);
        sweep.hueBy(0.2f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();
        c2.pause(duration-duration/4).queue(sweep).fadeOut(500).deleteWhenDone();
    }    


    public void radialOrange(){
        //ssm.playSound("002");
        ssm.playGroupRandom("6");
        // get location of fixture f01.
        // Point3d loc           = this.getFixture("f01").getLocation();
        // ReferenceDimension rd = this.getFixture("f01").getRealDimensions();

        int duration = 5000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("grad1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
        c2.pause(duration/2).queue(sweep).fadeOut(500).deleteWhenDone();    

    }

    public void radialRedMag(){
        ssm.playGroupRandom("6");

        int duration = 3000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_one_red_mag"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
    }

    public void fadeOrangeSlow(){
        ssm.playSound("Timpani_C2");

        int duration = 8000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("orange"), Color.getHSBColor(.4f, .99f, .99f), -width/2, 0, width, eam.getFrameDimensions().height, 1.0f);

        c1.pause(2500).fadeOut(duration-2500).deleteWhenDone();    
    }

    public void fadeBluePurpleSlow(){
        ssm.playSound("HornCombo_C2");

        int duration = 10000;
        Clip c1 = eam.addClip(eam.getContent("bluePurple"), Color.getHSBColor(.0f, .99f, .99f), 0, 20, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 0.0f);

        int pulseDur = 200;
        Sequence pulseIn = new Sequence();
        pulseIn.alphaTo(1.0f).duration(pulseDur);
        Sequence pulseOut = new Sequence();
        pulseOut.alphaTo(0.0f).duration(duration - pulseDur);

        //c1.pause(2500).fadeOut(duration-2500).deleteWhenDone();    
        c1.queue(pulseIn).queue(pulseOut).deleteWhenDone();    
    }

    public void radialBlueGreen3(){
        ssm.playGroupRandom("8");

        int duration = 6000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_three_blue_green"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("grad1200_three_blue_green"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
        c2.pause(duration/2).queue(sweep).fadeOut(500).deleteWhenDone();    
    }





    /** INTERACTIVE FEEDBACK CUES  ****************************/    


    public void randColor(Fixture fixture){
        red(fixture);
    }

    public void train(Fixture fixture) { 
        vertWavesRedMag();
    }

    public void red(Fixture fixture){

        ssm.playSound("002");

        Clip c = eam.addClip(eam.getContent("red"),
                (int)fixture.getLocation().x - 4,
                (int)fixture.getLocation().y - 4, 10, 10, 1.0f);

        Sequence huechange = new Sequence();
        huechange.hueBy(0.3f);

        c.queue(huechange).pause(800).fadeOut(1000).deleteWhenDone();
    }

    public void randomVibraSound(){

        //ssm.playSound(getRandVibra());
        ssm.playGroupRandom(chordIndex + "");
    }

    public void randomVibraTest(){

        //ssm.playSound(getRandVibra());

        ssm.playGroupRandom(chordIndex + "");
    }

    public void floraRand(Fixture fixture){

        randomVibraSound();
        iPulseVaseSensor();

        /* ranges
         * 0-0.2
         * .77-1.0
         */
        float hueMin = 0.7f;
        float hueDelta = 0.5f;
        float randHue = hueMin + (float)((Math.random() * hueDelta));
        //logger.info("RANDOM HUE = " + randHue);
        Clip c = eam.addClip(null, Color.getHSBColor(randHue, .99f, .99f),(int)fixture.getLocation().x - 4,(int)fixture.getLocation().y - 4, 10, 10, 1.0f);

        Sequence huechange = new Sequence();
        float hueChangeRange = 0.25f;
        float huernd = hueChangeRange - (float)(Math.random() * hueChangeRange * 2);
        //logger.info("Random hue change is " + huernd);
        huechange.hueBy(huernd);
        huechange.duration(2000);
        huechange.alphaTo(0.5f);
        c.queue(huechange).fadeOut(1000).deleteWhenDone();
    }

    public void redRandBlur(Fixture fixture){

        randomVibraSound();
        iPulseVaseSensor();

        int dia = 64;
        Clip c = eam.addClip(eam.getContent("blurDisc_32_red"),null,(int)fixture.getLocation().x - dia/2 + detOffset,(int)fixture.getLocation().y - dia/2 + detOffset, dia, dia, 1.0f);

        Sequence huechange = new Sequence();
        //float hueShift = 0.2f;
        float hueShift = 0.8f;
        float huernd = hueShift/2 - (float)(Math.random() * hueShift);
        //logger.info("Random hue change is " + huernd);
        huechange.hueBy(huernd).duration(2000);
        huechange.alphaTo(0.5f).duration(2000);
        c.queue(huechange).fadeOut(1000).deleteWhenDone();

    }

    public void redRandBlurAnim(Fixture fixture){

        randomVibraSound();
        iPulseVaseSensor();

        int dia = 64;
        Clip c = eam.addClip(eam.getContent("blurdisc_hue40"),null,(int)fixture.getLocation().x - dia/2 + detOffset,(int)fixture.getLocation().y - dia/2 + detOffset, dia, dia, 1.0f);

        Sequence huechange = new Sequence();
        //float hueShift = 0.2f;
        float hueShift = 0.8f;
        float huernd = hueShift/2 - (float)(Math.random() * hueShift);
        //logger.info("Random hue change is " + huernd);
        //huechange.hueBy(huernd).duration(2000);
        huechange.alphaTo(0.5f).duration(3000);
        c.queue(huechange).fadeOut(1000).deleteWhenDone();

    }

    public void redRandBlurAnimNeg(Fixture fixture){

        randomVibraSound();
        iPulseVaseSensor();

        int dia = 64;
        Clip c = eam.addClip(eam.getContent("blurdisc_hue-40"),null,(int)fixture.getLocation().x - dia/2 + detOffset,(int)fixture.getLocation().y - dia/2 + detOffset, dia, dia, 1.0f);

        Sequence huechange = new Sequence();
        //float hueShift = 0.2f;
        float hueShift = 0.8f;
        float huernd = hueShift/2 - (float)(Math.random() * hueShift);
        //logger.info("Random hue change is " + huernd);
        //huechange.hueBy(huernd).duration(2000);
        huechange.alphaTo(0.5f).duration(3000);
        c.queue(huechange).fadeOut(1000).deleteWhenDone();

    }





    public void green(Fixture fixture){

        randomVibraSound();

        Clip c = eam.addClip(eam.getContent("green"),
                (int)fixture.getLocation().x - 4,
                (int)fixture.getLocation().y - 4, 10, 10, 1.0f);

        c.pause(800).fadeOut(1000).deleteWhenDone();
    }








    /*************************/    


    private Fixture getFixture(String id){
        for (Fixture f : elu.getFixtures()){
            if (f.getName().equals(id)){
                return f;
            }
        }
        return null;
    }
}