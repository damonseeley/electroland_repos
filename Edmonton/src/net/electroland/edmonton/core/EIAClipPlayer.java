package net.electroland.edmonton.core;

import java.awt.Color;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;

import org.apache.log4j.Logger;

public class EIAClipPlayer {

    private AnimationManager anim;
    private ELUManager elu;
    protected Clip quiet, live;

    static Logger logger = Logger.getLogger(EIAClipPlayer.class);
    
    public EIAClipPlayer(AnimationManager am, ELUManager elu)
    {
        this.anim = am;
        this.elu = elu;

        // quiet = screen saver
        quiet = anim.addClip(new SolidColorContent(null), 0, 0, am.getStageDimensions().width, am.getStageDimensions().height, 0.0);
        // live = tracking users
        live = anim.addClip(new SolidColorContent(null), 0, 0, am.getStageDimensions().width, am.getStageDimensions().height, 1.0);
    }


    /*
     * EIA Live Show methods
     */
    int debugId = 0;
    public void localTrill4Up(double x) {
//        logger.info("localTrill4Up@ " + x);

        x = findNearestLight(x,true);
        //8 px wide
        int barWidth = 3;
        int barHeight = 16;
        int of = 1; //offset to hit lights

        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        // create a parent with NO color background
        Clip parent = live.addClip(new SolidColorContent(null), (int)x - barWidth * 2 - of, 0, barWidth * 4, 16, 1.0);

        //11 177 342 507 - new timing based on v18 sound
        Clip trill1 = parent.addClip(simpleClip2, -of, 0, barWidth, barHeight, 1.0);
        Clip trill2 = parent.addClip(simpleClip2, barWidth - of, 0, barWidth,  barHeight, 1.0, 166);
        Clip trill3 = parent.addClip(simpleClip2, 2 * barWidth - of, 0, barWidth, barHeight, 1.0, 331);
        Clip trill4 = parent.addClip(simpleClip2, 3 * barWidth - of, 0, barWidth, barHeight, 1.0, 496);
        
        //fade em all out
        // delete will kill the objecthe children automatically.
        parent.delay(1000).fadeOut(2000).delete();
        //parent.delay(3500).fadeOut(1000).delete();
        //parent.debug = debugId++;
    }
    
    public void s1v2TwoNote(double x){
      //logger.info("localStabA@ " + x);
        x = findNearestLight(x,true);
        double nextX = findNearestLight(x+3.5,true);
        //8 px wide
        int barWidth = 3;

        //create all bars, but at 0.0 alpha to popin later
        Content chord1 = new SolidColorContent(Color.WHITE);
        Content chord2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(chord1, (int)x-barWidth/2,0,barWidth,16, 1.0);
        Clip stab2 = live.addClip(chord2, (int)nextX-barWidth/2,0,barWidth,16, 1.0,168);

        //fade out
        stab1.delay(250).fadeOut(1500).delete();
        stab2.delay(250).fadeOut(1500).delete();
    }


    public void localStabSmall(double x) {
        //logger.info("localStabA@ " + x);
        x = findNearestLight(x,true);
        //8 px wide
        int barWidth = 3;

        //create all bars, but at 0.0 alpha to popin later
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0);

        //fade out
        stab1.delay(250).fadeOut(4000).delete();
    }

    public void localStabBig(double x) {
        //logger.info("localStabA@ " + x);
        x = findNearestLight(x,true);
        //8 px wide
        int barWidth = 6;

        //create all bars, but at 0.0 alpha to popin later
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0);

        //fade out
        stab1.delay(800).fadeOut(5000).delete();
    }

    /*
     * Globals
     */

    public void megaSparkleFaint(double x) {
        logger.info("global sparkle faint@ " + x);

        Content sparkleClip320 = anim.getContent("sparkleClip320");
        Clip faintSparkle = anim.addClip(sparkleClip320, 0,0,635,16, 0.0);

        //fadein, wait, fadeout
        Change lightFade = new LinearChange().alphaTo(.05);
        faintSparkle.delay(500).queueChange(lightFade, 4000).delay(12000).fadeOut(2000).delete();
    }
    
    
    /*
     * END EIA Live Show methods
     */



    /**
     * Test stuff
     */

    public void testClip(double x)
    {
        System.out.println("PLAY AT " + x);
    }

    public void sparkleClip32(double x){/*
        logger.debug("sparkleClip32 started at x=" + x);

        Content sparkleClip320 = anim.getContent("sparkleClip320");
        Clip clip = anim.addClip(sparkleClip320, (int)x-16, 0, 32, 16, 0.0);

        //fadein, wait, fadeout
        clip.fadeIn(500).delay(500).fadeOut(500).delete();*/
    }


    /**
     * Local Util Methods
     */

    private double findNearestLight(double x, boolean forward) {

        double closestX = -20;
        for (Fixture f: elu.getFixtures()) {
            if (Math.abs(x-f.getLocation().x) < Math.abs(x-closestX)) {
                closestX = f.getLocation().x;
            }
        }
        //logger.info("ClipPlayer: Track x= " + x + " & closest fixture x= " + closestX);
        return closestX;
    }

    private void LocalNoteTrill(double x, double width, double numNotes, int[] times, boolean LtoR){

    }

}