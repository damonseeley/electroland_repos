package net.electroland.norfolk.core;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.Map;

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

    public ClipPlayer(Animation eam, SimpleSoundManager ssm, ELUManager elu, ElectrolandProperties props){

        this.eam = eam;
        this.eam.addListener(this);
        this.ssm = ssm;
        this.elu = elu;
        this.ssm.load(props);
        this.configure(props);
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
        new ClipPlayerGUI(this);
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

    public void bounceSlowWave(Fixture fixture){

        ssm.playSound("001");

        Clip c = eam.addClip(eam.getContent("slowWave"),
                                (int)fixture.getLocation().x,
                                (int)fixture.getLocation().y, 100, 100, 1.0f);

        Sequence bounce = new Sequence();

        bounce.yTo(150).yUsing(new QuinticIn())
              .xBy(100).xUsing(new Linear())
              .scaleWidth(2.0f)
              .duration(1000)
       .newState()
              .yTo(75).yUsing(new CubicOut())
              .xBy(100).xUsing(new Linear())
              .scaleWidth(.5f)
              .duration(1000);

       c.queue(bounce).queue(bounce).queue(bounce).fadeOut(500).deleteWhenDone();
    }

    public void sweepWhiteDown(){
        ssm.playSound("002");

        // get location of fixture f01.
        Point3d loc           = this.getFixture("f01").getLocation();
        ReferenceDimension rd = this.getFixture("f01").getRealDimensions();
        System.out.println("f01 is at: " + loc + " of dimensions " + rd);

        int height = 50;
        Clip c = eam.addClip(Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);
//        Clip c = eam.addClip(eam.getContent("whitegradient"), Color.getHSBColor(.9f, .8f, .7f), 0, -height, eam.getFrameDimensions().width, height, 1.0f);

        Sequence sweep = new Sequence();
        sweep.yTo(eam.getFrameDimensions().height).duration(15000);
        sweep.hueBy(1.0f);
        sweep.brightnessTo(0.0f);

        c.queue(sweep).fadeOut(500).deleteWhenDone();
    }

    public void red(Fixture fixture){

        ssm.playSound("002");

        Clip c = eam.addClip(eam.getContent("red"),
                                (int)fixture.getLocation().x - 10,
                                (int)fixture.getLocation().y - 10, 20, 20, 1.0f);

        c.fadeOut(1000).deleteWhenDone();
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