package net.electroland.edmonton.core;

import java.awt.Color;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.lighting.ELUManager;

import org.apache.log4j.Logger;

public class EIAClipPlayer2 extends EIAClipPlayer {

    //private SoundController sc;

    static Logger logger = Logger.getLogger(EIAClipPlayer2.class);

    public EIAClipPlayer2(AnimationManager am, ELUManager elu, SoundController sc)
    {
        super(am, elu, sc);
    }

    private double barOff = -3.69;
    private double lookAhead = -3.2;
    private int topBar = 5;
    private int bottomBar = 8;
    private int barHeight = 2;

    
    
    public void vertSixFill(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int dOff = 150; //delay offset
        int startX = (int)(x-barWidth/2);

        Clip top1 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*0),topBar,barWidth,barHeight, 1.0); 
        Clip bottom1 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*0),bottomBar,barWidth,barHeight, 1.0, dOff*1);
        Clip top2 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*1),topBar,barWidth,barHeight, 1.0, dOff*2); 
        Clip bottom2 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*1),bottomBar,barWidth,barHeight, 1.0, dOff*3); 
        Clip top3 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*2),topBar,barWidth,barHeight, 1.0, dOff*4); 
        Clip bottom3 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*2),bottomBar,barWidth,barHeight, 1.0, dOff*5); 

        top1.delay(350).fadeOut(350).delete();
        bottom1.delay(350).fadeOut(350).delete();
        top2.delay(350).fadeOut(350).delete();
        bottom2.delay(350).fadeOut(350).delete();
        top3.delay(350).fadeOut(350).delete();
        bottom3.delay(350).fadeOut(350).delete();

        sc.playSingleChannelBlind("entrance6.wav", x, 0.4f);

    }

    public void grow(double x){
        logger.info("grow");

        int barWidth = 3;
        int dist = 65;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip grow = live.addClip(simpleClip2,(int)x-barWidth,topBar,barWidth,barHeight*3, 1.0);

        Change out = new LinearChange().xTo(x-dist).widthTo(dist);
        Change back = new LinearChange().xTo(x+lookAhead*1).widthTo(barWidth);

        grow.queueChange(out,600).queueChange(back,600).fadeOut(100).delete();
        sc.playSingleChannelBlind("highlight_22.wav", x, 1.0f);
    }


    public void entryWavePM1(double x) {
        //logger.info("bigWaveAll@ " + x);
        x = findNearestLight(x+lookAhead,true);

        Content waveImage = anim.getContent("waveImage");

        int waveWidth = 24;
        Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
        waveImageClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(260);
        waveImageClip.queueChange(waveMove, 3000).delete();
        //one.delay(4000).queueChange(change6, 1000);
        //faintSparkle.delay(500).queueChange(lightFade, 4000).delay(12000).fadeOut(2000).delete();

        sc.playSingleChannelBlind("kotu_04.wav", x, 1.0f);

    }

    public void entryWavePM2(double x) {
        //logger.info("bigWaveAll@ " + x);
        x = findNearestLight(x+lookAhead,true);

        Content waveImage = anim.getContent("waveImage");

        int waveWidth = 24;
        Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
        waveImageClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(-waveWidth);
        waveImageClip.queueChange(waveMove, 3000).delete();

        sc.playSingleChannelBlind("kotu_04.wav", x, 1.0f);

    }

    public void boomerang(double x){
        logger.info("boomerang");

        int barWidth = 18;
        int dist = 65;

        x = findNearestLight(x+lookAhead/2,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip boomerang = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight*3, 1.0);

        Change out = new LinearChange().xTo(x-dist).scaleWidth(1.5);
        Change back = new LinearChange().xTo(x+lookAhead*2).scaleWidth(0.66).alphaTo(0.1);

        boomerang.queueChange(out, 500).queueChange(back,500).fadeOut(100).delete();
            
        sc.playSingleChannelBlind("vert_disconnect_long_whoosh03.wav", x, 1.0f);
    }


    public void randomBars(double x){
        int maxLoop = 14;
        int minLoop = 7;
        int loop = (int)(maxLoop*Math.random());
        int delay = 120;
        boolean top = true;
        for (int l=0; l<Math.max(loop,minLoop); l++){
            if (top){
                topRandomBar(x,delay*l,6,-32);
                top = false;
            } else {
                bottomRandomBar(x,delay*l,6,-32);
                top = true;
            }
        }
        
        sc.playSingleChannelBlind("vert_connect_med_whoosh16_long.wav", x, 1.0f);
    }
    




    public void smVertDoublet(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 9;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2,bottomBar,barWidth,barHeight, 1.0, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //now play a sound!
        sc.playSingleChannelBlind("lumen_entrance7.wav", x, 1.0f);

    }

    public void bigVertDoublet(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 32;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth+2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 1.0, 1000); 

        topBlip.delay(600).fadeOut(600).delete();
        bottomBlip.delay(800).fadeOut(600).delete();

        //now play a sound!
        // something bigger here
        sc.playSingleChannelBlind("piano_doublet01.wav", x, 1.0f);

    }
    
    
    public void randomBarsMore(double x){
        int maxLoop = 16;
        int minLoop = 8;
        int loop = (int)(maxLoop*Math.random());
        int delay = 120;
        boolean top = true;
        for (int l=0; l<Math.max(loop,minLoop); l++){
            if (top){
                topRandomBar(x,delay*l,7,-48);
                top = false;
            } else {
                bottomRandomBar(x,delay*l,7,-48);
                top = true;
            }
        }
        
        //vert_connect_med_whoosh16.wav
        sc.playSingleChannelBlind("vert_connect_med_whoosh16_long.wav", x, 1.0f);
    }
    
    private int randBarSpeed = 1500;

    public void topRandomBar(double x, int delay, int maxBarLength, int barDest){

        logger.info("topRandomBar");

        int barWidth = (int)(maxBarLength*Math.random())+1;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight, 1.0, delay);

        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(250).delete();
    }

    public void bottomRandomBar(double x, int delay, int maxBarLength, int barDest){

        logger.info("bottomRandomBar");

        int barWidth = (int)(maxBarLength*Math.random())+1;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,bottomBar,barWidth,barHeight, 1.0, delay);

        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(250).delete();
    }

    public void blip2(double x) {

        boolean doBlip = true;
        boolean doBlipSound = true;

        if (doBlip){
            x = findNearestLight(x+lookAhead,true);
            //orig int barWidth = 3;
            int barWidth = 3;
            Content simpleClip2 = new SolidColorContent(Color.WHITE);
            Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 0.8); //set the alpha to 0.5 to get 50% brightness on creation
            blip1.delay(200).fadeOut(400).delete();

            //now play a sound

            sc.playSingleChannelBlind("marimba_mid_01.wav", x, 1.0f); // the firs test, kind of "crunch" sound

        }
    }








}