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

    public void localTrill4Up(double x) {
//        logger.info("localTrill4Up@ " + x);

        x = findNearestLight(x,true);
        //8 px wide
        int barWidth = 3;
        int of = 1; //offset to hit lights

        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip trill1 = live.addClip(simpleClip2, (int)x - barWidth * 2 - of, 0, barWidth,     16, 1.0, 0);
        Clip trill2 = live.addClip(simpleClip2, (int)x - barWidth - of, 0, barWidth,         16, 1.0, 170);
        Clip trill3 = live.addClip(simpleClip2, (int)x - of, 0, barWidth,                     16, 1.0, 375);
        Clip trill4 = live.addClip(simpleClip2, (int)x + barWidth - of, 0, barWidth,         16, 1.0, 530);

        //fade em all out
        Change fadeOut = new LinearChange().alphaTo(0.01);
        trill1.delay(3500).queueChange(fadeOut, 500).delete();
        trill2.delay(3500).queueChange(fadeOut, 500).delete();
        trill3.delay(3500).queueChange(fadeOut, 500).delete();
        trill4.delay(3500).queueChange(fadeOut, 500).delete();

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
        stab1.delay(800).fadeOut(5000).delete();
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
        Change lightFade = new LinearChange().alphaTo(.2);
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