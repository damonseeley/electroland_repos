package net.electroland.norfolk.core;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
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
import net.electroland.eio.Coordinate;
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
    private Map<String, Method> sensorToClips;

    public ClipPlayer(Animation eam, SimpleSoundManager ssm, ELUManager elu, ElectrolandProperties props){

        this.eam = eam;
        this.eam.addListener(this);
        this.ssm = ssm;
        this.elu = elu;
        this.ssm.load(props);
        this.configure(props);
    }

    public void configure(ElectrolandProperties props){

        sensorToClips = new HashMap<String, Method>();

        for (ParameterMap mappings : props.getObjects("channelClip").values()){
            String channelId = mappings.get("channel");
            try {
                Method method = this.getClass().getMethod(mappings.get("clipPlayerMethod"), Coordinate.class);
                sensorToClips.put(channelId, method);
                logger.info("mapped channel " + channelId + " to " + method.getName());
            } catch (NoSuchMethodException e) {
                e.printStackTrace();
            }

        }
    }

    public void play(InputChannel channel){

        try {

            Method m = sensorToClips.get(channel.getId());

            if (m != null){
                logger.debug("clipPlayer.play " + m.getName() + " at " + channel.getLocation());
                m.invoke(this, channel.getLocation());
            }else{
                logger.debug("no method defined for channel " + channel.getId());
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

    public void bounceSlowWave(Coordinate location){

        ssm.playSound("001");

        Clip c = eam.addClip(eam.getContent("slowWave"),
                                (int)location.getX(),
                                (int)location.getY(), 100, 100, 1.0f);

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

    public void sweepWhiteDown(Coordinate location){
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

    public void red(Coordinate location){

        ssm.playSound("002");

        Clip c = eam.addClip(eam.getContent("red"),
                                (int)location.getX(),
                                (int)location.getY(), 100, 100, 1.0f);
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
}