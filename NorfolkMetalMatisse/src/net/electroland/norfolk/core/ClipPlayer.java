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

import javax.vecmath.Point3d;

import net.electroland.ea.Animation;
import net.electroland.ea.AnimationListener;
import net.electroland.ea.Clip;
import net.electroland.ea.Sequence;
import net.electroland.ea.easing.CubicOut;
import net.electroland.ea.easing.Linear;
import net.electroland.ea.easing.QuinticIn;
import net.electroland.eio.InputChannel;
import net.electroland.norfolk.sound.SimpleSoundManager;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.ParameterMap;
import net.electroland.utils.ReferenceDimension;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;

import org.apache.log4j.Logger;



public class ClipPlayer implements AnimationListener {

    private static Logger logger = Logger.getLogger(ClipPlayer.class);
    private Animation eam;
    private SimpleSoundManager ssm;
    private ELUManager elu;
    private Map<String, Target> sensorToClips;
    private Collection<Method> globalClips;
    
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
        	logger.info("Changed chordIndex to " + chordIndex);
        }
    }

    public void configure(ElectrolandProperties props){

        sensorToClips = new HashMap<String, Target>();

        for (ParameterMap mappings : props.getObjects("channelClip").values()){
            String channelId = mappings.get("channel");
            try {
                Method method = this.getClass().getMethod(mappings.get("clipPlayerMethod"), Fixture.class);
                String fixtureId = mappings.get("fixture");
                Fixture fixture = this.getFixture(fixtureId); // can be null
                sensorToClips.put(channelId, new Target(fixture, method));
                logger.info("mapped channel " + channelId + " to " + method.getName() + " on fixture " + fixture);
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }
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

    public void play(InputChannel channel){

        try {

            Target t = sensorToClips.get(channel.getId());

            if (t != null && t.method != null && t.fixture != null){
                logger.debug("clipPlayer.play " + t.method.getName() + " at " + t.fixture + " for " + channel);
                t.method.invoke(this, t.fixture);
            }

        } catch (SecurityException e) {
            e.printStackTrace();
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.getTargetException().printStackTrace();
        }
    }
    
    /** ANIMATIONS ****************************/

    public void sweepWhiteDown(){

        int height = 120;
        height = (int)(Math.random() * 100 + 70);
        //Clip c = eam.addClip(Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(3000);
        sweep.hueBy(1.0f);
        sweep.brightnessTo(0.5f);

        c.queue(sweep).fadeOut(500).deleteWhenDone();
    }
    
    public void vertSweeps(){
        
        int duration = 3000;

        int height = 120;
        //Clip c = eam.addClip(Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c1 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.4f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c2 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.5f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c3 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.6f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
        Clip c4 = eam.addClip(eam.getContent("gradientinvert"), Color.getHSBColor(.7f, .99f, .99f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(duration);
        sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);
        

        c1.queue(sweep).fadeOut(500).deleteWhenDone();
        c2.pause(900).queue(sweep).fadeOut(500).deleteWhenDone();
        c3.pause(1800).queue(sweep).fadeOut(500).deleteWhenDone();
        c4.pause(2700).queue(sweep).fadeOut(500).deleteWhenDone();
               
    }
        
    public void vertWavesRedMag(){
        //ssm.playSound("002");
        
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
    
    
    public void timedShow() {
    	// conditional here to randomize showing
    	int rand = (int)(Math.random() * 100);
    	logger.info(rand);
    	if (rand < 20) {
    		radialOrange();
    	} else if (rand < 40) {
    		radialBlueGreen3();
    	} else if (rand < 60) {
    		radialRedMag();
    	} else if (rand < 80){
    		fadeOrangeSlow();
    	} else {
    		vertWavesRedMag();
    	}
    	
    	
    
    	
    	
    }
    
    public void radialOrange(){
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
        int duration = 8000;
        int width = 600;
        Clip c1 = eam.addClip(eam.getContent("orange"), Color.getHSBColor(.4f, .99f, .99f), -width/2, 0, width, eam.getFrameDimensions().height, 1.0f);

        Sequence sweep = new Sequence();
        //sweep.xTo(eam.getFrameDimensions().width + width).duration(duration);
        //sweep.hueBy(0.2f);
        //sweep.brightnessTo(0.5f);
        
        c1.queue(sweep).pause(2500).fadeOut(duration-2500).deleteWhenDone();    
    }
    
    public void radialBlueGreen3(){
        //ssm.playSound("002");
        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);
        
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
   
    
    
    /** LOCAL FIXTURE-SPECIFIC ANIMATIONS **/

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
    
    
    
    

    @Override
    public void messageReceived(Object message) {
        // TODO Auto-generated method stub
        // animation manager
    }

    private Fixture getFixture(String id){
        for (Fixture f : elu.getFixtures()){
            if (f.getName().equals(id)){
                return f;
            }
        }
        return null;
    }

    class Target {

        public Fixture fixture;
        public Method method;

        public Target(Fixture fixture, Method method){
            this.fixture = fixture;
            this.method = method;
        }
    }
}