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
    private Clip screensaver;

    private Timer chordTimer;
    int chordIndex;
    int chordIndexMax;
    long chordDur;

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
        screensaver = eam.addClip(null, null, 0, 0, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 1.0f);
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
                    method.invoke(this);
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

    public void play(String clipName, InputChannel channel){

        Fixture f = channelToFixture.get(channel.getId());

        try {

            Method method = this.getClass().getMethod(clipName, Fixture.class);
            if (f == null){
                method.invoke(this);
            }else{
                method.invoke(this, f);
            }
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e){
            e.printStackTrace();
        }
    }
    
    

    /** SCREENSAVERS ****************************/
    
    
    public void screensaverBlueClouds(){

        int duration   = 3000;
        int height     = 800;

        Clip clouds    = screensaver.addClip(eam.getContent("clouds_200x800_blue"), 
                                             Color.getHSBColor(.0f, .0f, .0f), 
                                             0, -height, 
                                             eam.getFrameDimensions().width, height, 
                                             0.5f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(duration);

        clouds.queue(sweep).announce(screensaver).deleteWhenDone();
    }

    public void enterScreensaverMode(int millis){
        screensaver.fadeIn(millis);
    }

    public void exitScreensaverMode(int millis){
        screensaver.fadeOut(millis);
    }

    /**
     * When screensavers complete, this gets called. This is the chance
     * to decide what screensaver to play next.
     */
    @Override
    public void messageReceived(Object message) {
        if (message == screensaver){
            // this is where you choose the next screensaver or keep the current
            // one cycling.
            screensaverBlueClouds();
        }
    }
    
    
    
    
    

    /** ACCENT SHOWS FOR TRIPLETS AND THE LIKE ****************************/
    
    
    public void radialCobrasOrange(){

        int duration = 3000;
        int width = 600;
        int vLoc = 176; // start of cobras
        int vHeight = 22; // cover all cobras
       
        Clip parent = eam.addClip(null, null, -width*2, vLoc, width*2, vHeight, 1.0f);
        parent.addClip(eam.getContent("bar1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), 0, 0, width, vHeight, 1.0f);
        parent.addClip(eam.getContent("bar1200_one_org"), Color.getHSBColor(.4f, .99f, .99f), width, 0, width, vHeight, 1.0f);
        
        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width).duration(duration);

        parent.queue(sweep).deleteWhenDone(); 
        
    }
    
    
    
    
    /** BIG SHOWS ****************************/


    public void vertWavesRedMag(){
        ssm.playGroupRandom("6");

        int duration = 5000;

        int height = 600;
        //Clip c = eam.addClip(Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c1 = eam.addClip(eam.getContent("grad1200_vert_three_red_mag"), Color.getHSBColor(.4f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("grad1200_vert_three_red_mag"), Color.getHSBColor(.5f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);        //Clip c4 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.7f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);

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
        //sweep.brightnessTo(0.5f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
        c2.pause(duration/2).queue(sweep).fadeOut(500).deleteWhenDone();    

    }

    public void radialBlueGreen(){
        ssm.playGroupRandom("6");

        int duration = 5000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_one_blue_green"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);

        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
    }

    public void radialRedMag(){
        ssm.playGroupRandom("6");

        int duration = 3000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_one_red_mag"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);
        
        c1.queue(sweep).fadeOut(500).deleteWhenDone();    
    }

    public void radialRedMagSlow(){

        int duration = 24000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_one_red_mag"), Color.getHSBColor(.4f, .99f, .99f), -width/2, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);
        
        c1.queue(sweep).fadeOut(500).deleteWhenDone();
    }

    public void fadeOrangeSlow(){
        ssm.playGroupRandom("6");

        int duration = 8000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("orange"), Color.getHSBColor(.4f, .99f, .99f), -width/2, 0, width, eam.getFrameDimensions().height, 1.0f);

        c1.pause(2500).fadeOut(duration-2500).deleteWhenDone();    
    }
    
    public void fadeBluePurpleSlow(){
        ssm.playSound("HornCombo_C2");

        int duration = 10000;
        int width = 200;
        Clip c1 = eam.addClip(eam.getContent("bluePurple"), Color.getHSBColor(.0f, .99f, .99f), 0, 0, eam.getFrameDimensions().width, eam.getFrameDimensions().height, 1.0f);

        c1.pause(2500).fadeOut(duration-2500).deleteWhenDone();    
    }

    public void radialBlueGreen3(){
    	ssm.playGroupRandom("6");
    	
        int duration = 6000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("grad1200_three_blue_green"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("grad1200_three_blue_green"), Color.getHSBColor(.4f, .99f, .99f), -width, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);

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
        //sweep.brightnessTo(0.5f);
        
        c.queue(huechange).pause(800).fadeOut(1000).deleteWhenDone();
    }

    public void randomVibra(){

        //ssm.playSound(getRandVibra());
        ssm.playGroupRandom(chordIndex + "");
    }

    public void randomVibraTest(){

        //ssm.playSound(getRandVibra());
        
        ssm.playGroupRandom(chordIndex + "");
    }

    public void redRand(Fixture fixture){

        randomVibra();

        Clip c = eam.addClip(null,new Color(255,0,0),(int)fixture.getLocation().x - 4,(int)fixture.getLocation().y - 4, 10, 10, 1.0f);

        Sequence huechange = new Sequence();
        float huernd = 0.1f - (float)(Math.random() *0.2f);
        logger.info("Random hue change is " + huernd);
        huechange.hueBy(huernd);
        huechange.duration(2000);
        huechange.brightnessTo(0.5f);
        c.queue(huechange).fadeOut(1000).deleteWhenDone();
        
    }
    
    public void green(Fixture fixture){

        randomVibra();

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