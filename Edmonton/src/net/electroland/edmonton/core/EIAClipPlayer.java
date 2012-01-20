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

    int cWidth = 635;
    int cHeight = 16;
    int pmvr2Start = 240;

    int wNote = 659; //ms
    int hNote = wNote/2;
    int qNote = wNote/4; //ms per qnote
    int eNote = wNote/8;

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

    public void localTrill4Down(double x) {
        logger.info("localTrill4Down@ " + x);

        double x1 = (int)findNearestLight(x,true);
        int barOff = 4;
        double x2 = (int)findNearestLight(x1-barOff,true);
        double x3 = (int)findNearestLight(x2-barOff,true);
        double x4 = (int)findNearestLight(x3-barOff,true);
        int barWidth = 3;

        int of = 1; //offset to hit lights

        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip trill1 = live.addClip(simpleClip2, (int)x1-of, 0, barWidth, cHeight, 1.0);
        Clip trill2 = live.addClip(simpleClip2, (int)x2-of, 0, barWidth,  cHeight, 1.0, 166);
        Clip trill3 = live.addClip(simpleClip2, (int)x3-of, 0, barWidth, cHeight, 1.0, 331);
        Clip trill4 = live.addClip(simpleClip2, (int)x4-of, 0, barWidth, cHeight, 1.0, 496);
        //fade em all out
        trill1.delay(1000).fadeOut(2000).delete();
        trill2.delay(1000).fadeOut(2000).delete();
        trill3.delay(1000).fadeOut(2000).delete();
        trill4.delay(1000).fadeOut(2000).delete();


    }

    public void twoNoteChord(double x){
        logger.info("twoNoteChord@ " + x);
        int barOff = 4;
        int barWidth = 3;
        int of = 1; //offset to hit lights
        int x1 = (int)findNearestLight(x,true);
        int x2 = (int)findNearestLight(x-barOff,true);

        Content chord1 = new SolidColorContent(Color.WHITE);
        Content chord2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(chord1, x1-of,0,barWidth,16, 1.0);
        Clip stab2 = live.addClip(chord2, x2-of,0,barWidth,16, 1.0,168);
        //fade out
        stab1.delay(250).fadeOut(1500).delete();
        stab2.delay(250).fadeOut(1500).delete();
    }


    public void localStabSmall(double x) {
        logger.info("localStabSmall@ " + x);
        x = findNearestLight(x,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0);
        //fade out
        stab1.delay(250).fadeOut(3000).delete();
    }

    public void localStabSmallLong(double x) {
        //same but long fadeout
        logger.info("localStabSmallLong@ " + x);
        x = findNearestLight(x,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,cHeight, 1.0);
        //fade out
        stab1.delay(250).fadeOut(8000).delete();
    }

    public void localStabExpand(double x) {
        logger.info("localStabBig@ " + x);
        x = findNearestLight(x,true);
        int barWidth = 8;
        int of = 1;
        //create all bars, but at 0.0 alpha to popin later
        Content blurSquare = anim.getContent("blurSquare64grad");
        Clip stab1 = live.addClip(blurSquare, (int)x-barWidth/2-of,0,barWidth,cHeight, 1.0);
        double newScale = 12.0;
        Change scale = new LinearChange().scaleWidth(newScale).xTo(x-barWidth/2*newScale-of).alphaTo(0.0);
        stab1.queueChange(scale, 800);
    }



    public void localStabBig(double x) {
        logger.info("localStabBig@ " + x);
        x = findNearestLight(x,true);
        int barWidth = 6;
        //create all bars, but at 0.0 alpha to popin later
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip stab1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0);
        //fade out
        stab1.delay(800).fadeOut(5000).delete();
    }


    //could be more generalized
    public void harpTrillUp(double x) {
        logger.info("HarpTrill @" + x);
        int barWidth = 3;

        int of = 1;
        x = findNearestLight(x-24,true);
        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        int ix = (int)x;

        Clip harp1 = live.addClip(simpleClip2, ix-barWidth*4-barWidth/2-of, 0, barWidth, cHeight, 1.0, 0);
        Clip harp2 = live.addClip(simpleClip2, ix-barWidth*3-barWidth/2-of, 0, barWidth, cHeight, 0.9, eNote);
        Clip harp3 = live.addClip(simpleClip2, ix-barWidth*2-barWidth/2-of, 0, barWidth, cHeight, 0.7, eNote*2);
        Clip harp4 = live.addClip(simpleClip2, ix-barWidth*1-barWidth/2-of, 0, barWidth, cHeight, 1.0, eNote*3);      
        Clip harp5 = live.addClip(simpleClip2, ix+barWidth*0-barWidth/2-of, 0, barWidth, cHeight, 0.8, eNote*4);
        Clip harp6 = live.addClip(simpleClip2, ix+barWidth*1-barWidth/2-of, 0, barWidth, cHeight, 0.9, eNote*5);
        Clip harp7 = live.addClip(simpleClip2, ix+barWidth*2-barWidth/2-of, 0, barWidth, cHeight, 0.9, eNote*6);
        Clip harp8 = live.addClip(simpleClip2, ix+barWidth*3-barWidth/2-of, 0, barWidth, cHeight, 1.0, eNote*7);

        harp1.delay(500).fadeOut(1000).delete();
        harp2.delay(500).fadeOut(1000).delete();
        harp3.delay(500).fadeOut(1000).delete();
        harp4.delay(500).fadeOut(1000).delete();
        harp5.delay(500).fadeOut(1000).delete();
        harp6.delay(500).fadeOut(1000).delete();
        harp7.delay(500).fadeOut(1000).delete();
        harp8.delay(500).fadeOut(1000).delete();
    }

    /*
     * Globals
     */

    public void harpFill(double x) {
        logger.info("HarpTrill @" + x);
        int barWidth = 3;

        int of = 1;
        x = findNearestLight(x-24,true);
        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int ix = (int)x;
        int xLow = 0;
        int xHigh = cWidth;
        int plucks = 8;
        int pluckDensity = 12;
        int pDelay = qNote;

        for (int p=0; p<plucks; p++){
            for (int d=0; d<pluckDensity; d++){
                int newx = (int)(Math.random()*(xHigh-xLow));
                live.addClip(simpleClip2, newx-of, 0, barWidth, cHeight, 1.0, p*pDelay).delay(400+(int)(Math.random()*400)).fadeOut(800).delete();
            }
        }
    }

    public void wholeNoteBeatRest(double x) {
        logger.info("HarpTrill @" + x);
        int barWidth = 3;

        int of = 1;
        x = findNearestLight(x-24,true);
        //create all bars, with each appearing in delayed intervals
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int beats = 16;
        int pDelay = wNote*2;

        for (int p=0; p<beats; p++){
            live.addClip(simpleClip2, 0, 0, cWidth, cHeight, 0.2, p*pDelay).fadeOut(wNote*2).delete();
        }
    }

    public void introSparkle(double x) {
        logger.info("introSparkle@ " + x);

        Content sparkleClipFast = anim.getContent("sparkleClipFast");
        Clip faintSparkle = live.addClip(sparkleClipFast, 0,0,cWidth,16, 0.0);
        faintSparkle.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change lightFade = new LinearChange().alphaTo(.15);
        faintSparkle.delay(0).queueChange(lightFade, 1000).delay(18500).fadeOut(2000).delete();
    }

    public void s1v1sparkle(double x) {
        logger.info("s1v1sparkle@ " + x);

        Content sparkleClipFast = anim.getContent("sparkleClipFast");
        Clip faintSparkle = live.addClip(sparkleClipFast, 0,0,cWidth,16, 0.0);
        faintSparkle.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change lightFade = new LinearChange().alphaTo(.15);
        faintSparkle.delay(0).queueChange(lightFade, 800).delay(19000).fadeOut(2000).delete();
    }

    public void screenSaverSparkle(double x) {
        logger.info("screenSaverSparkle@ " + x);
        // use x as total time here
        int tFadeIn = 2000;
        int tFadeOut = 2000;
        int remain = (int) (x - tFadeOut - tFadeIn);

        Content sparkleClipFast = anim.getContent("sparkleClipSaver");
        Clip faintSparkle = quiet.addClip(sparkleClipFast, 0,0,cWidth,16, 0.0);
        faintSparkle.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change lightFade = new LinearChange().alphaTo(.15);
        faintSparkle.queueChange(lightFade, tFadeIn).delay(remain).fadeOut(tFadeOut).delete();
    }

    public void screenSaverSparkle2(double x) {
        logger.info("screenSaverSparkle2@ " + x);
        // use x as total time here as a hack
        int tFadeIn = 2000;
        int tFadeOut = 2000;
        int remain = (int) (x - tFadeOut - tFadeIn);

        Content sparkleClipFast = anim.getContent("sparkleClipSaver2");
        Clip faintSparkle = quiet.addClip(sparkleClipFast, 0,0,cWidth,16, 0.0);
        faintSparkle.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change lightFade = new LinearChange().alphaTo(0.35);
        faintSparkle.queueChange(lightFade, tFadeIn).delay(remain).fadeOut(tFadeOut).delete();
    }





    public void blockWaveAll(double x) {
        logger.info("blockWaveAll@ " + x);
        Content waveBlock = new SolidColorContent(Color.WHITE);

        int waveWidth = 32;
        Clip waveClip = live.addClip(waveBlock, 260,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
        waveClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(-waveWidth);
        waveClip.queueChange(waveMove, 5000).delay(500).delete();
        //one.delay(4000).queueChange(change6, 1000);
        //faintSparkle.delay(500).queueChange(lightFade, 4000).delay(12000).fadeOut(2000).delete();
    }

    public void bigWaveAll(double x) {
        logger.info("bigWaveAll@ " + x);
        Content waveImage = anim.getContent("waveImage");

        int waveWidth = 32;
        Clip waveImageClip = live.addClip(waveImage, 235,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
        waveImageClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(-waveWidth);
        waveImageClip.queueChange(waveMove, 10000).delay(500).delete();
        //one.delay(4000).queueChange(change6, 1000);
        //faintSparkle.delay(500).queueChange(lightFade, 4000).delay(12000).fadeOut(2000).delete();
    }

    public void megaWaveDouble(double x) {
        logger.info("megaWaveDouble@ " + x);
        Content waveImage = anim.getContent("megaWave");

        int waveWidth = 256;
        int wave1End = 340;
        //pmvr2Start
        Clip wave1 = live.addClip(waveImage, cWidth-15,0,waveWidth/2,cHeight, 1.0); //add it as 32px wide at the end of the stage
        Clip wave2 = live.addClip(waveImage, pmvr2Start-8,0,waveWidth/3,cHeight, 1.0); //add it as 32px wide at the end of the stage
        wave1.zIndex = -100; // sets to far background
        wave2.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove1 = new LinearChange().xTo(wave1End-waveWidth).scaleWidth(3.0).alphaTo(0.0);
        Change waveMove2 = new LinearChange().xTo(0-waveWidth).scaleWidth(3.0).alphaTo(0.0);

        wave1.queueChange(waveMove1, 3500).delete();
        wave2.queueChange(waveMove2, 3500).delete();
    }


    public void randomWaves(double x) {
        logger.info("randomWaves @" + x);
        int xLow = 0;
        int xHigh = cWidth;
        int waveWidth = 32;

        //here we are using the x value as the number of waves to make
        int waves = (int)x;
        int pDelay = eNote;
        Content waveImage = anim.getContent("waveImage");
        for (int p=0; p<waves; p++){
            int newx = (int)(Math.random()*(xHigh-xLow));
            Change moveIt = new LinearChange().xBy(-16).alphaTo(0.0);
            live.addClip(waveImage, newx, 0, waveWidth, cHeight, 1.0, p*pDelay).queueChange(moveIt, 1500);
        }
    }


    /*
     * END EIA Live Show methods
     */




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