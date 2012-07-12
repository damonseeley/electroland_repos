package net.electroland.edmonton.core;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.LinkedHashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;
import java.util.Timer;
import java.util.TimerTask;


import net.electroland.ea.AnimationManager;
import net.electroland.ea.Change;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.changes.LinearChange;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.edmonton.core.EIAMainConductor.startupTests;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;
import net.electroland.utils.lighting.canvas.ELUCanvas2D;

import org.apache.log4j.Logger;

public class EIAClipPlayer2 {

    static Logger logger = Logger.getLogger(EIAClipPlayer2.class);
    protected AnimationManager anim;
    protected ELUManager elu;
    protected ELUCanvas2D canvas;
    protected SoundController sc;
    protected Clip quiet, live;
    protected TrafficFlowAnalyzer tfa;
    protected ElectrolandProperties propsGlobal;

    protected Timer phantomTimer;

    //spatial vars, specific to this exact canvas size and arrangement
    private double barOff = -3.69;
    private double lookAhead = -3.2;
    private int topBar = 5;
    private int bottomBar = 8;
    private int barHeight = 2;

    //from ClipPlayer1, to make waves work
    int cWidth = 635;
    int cHeight = 16;
    int pmvr2Start = 240;

    int wNote = 659; //ms
    int hNote = wNote/2;
    int qNote = wNote/4; //ms per qnote
    int eNote = wNote/8;


    public EIAClipPlayer2(Map<String, Object> context)
    {
        try{
            this.anim   = (AnimationManager)context.get("anim");
            this.elu    = (ELUManager)context.get("elu");
            this.sc     = (SoundController)context.get("soundController");
            this.canvas = (ELUCanvas2D)context.get("canvas");  
            this.tfa    = (TrafficFlowAnalyzer)context.get("tfa");
            this.propsGlobal = (ElectrolandProperties)context.get("propsGlobal");
        }catch(NullPointerException e){            
            logger.error(e);
            System.exit(-1);
        }catch(ClassCastException e){
            logger.error(e);
            System.exit(-1);
        }

        //propsGlobal.getRequiredBoolean("traffic", "phantoms", "low")
        phantomTimer = new Timer();
        if (Boolean.parseBoolean(propsGlobal.getRequired("traffic", "phantoms", "enabled"))){
            phantomTimer.schedule(new phantomTimerTask(), 1000);
        }
        live = anim.addClip(new SolidColorContent(null), 0, 0, anim.getStageDimensions().width, anim.getStageDimensions().height, 1.0);
        //logger.info("ClipPlayer2 created: " + this);
    }

    private class phantomTimerTask extends TimerTask {
        public void run() {
            // fire phantom
            int newTime = (int)(Math.random()*400)+200;
            phantomTimer.schedule(new phantomTimerTask(), newTime);

            if (tfa.getPM1Flow() == 0 && tfa.getPM2Flow() == 0) {
                //perhaps randomize number of firings
                accentStrobeRand(Math.random() * 620);
                //logger.info(this + " fired a phantom clip and newTime= " + newTime);
            }
        }
    }

    public void playClip(String name, double loc){

        try {

            logger.debug("Running clipPlayer2." + name + '(' + loc + ')');
            Method m = this.getClass().getMethod(name, double.class);
            m.invoke(this, loc);

        } catch (SecurityException e) {
            e.printStackTrace();
        } catch (NoSuchMethodException e) {
            e.printStackTrace();
        } catch (IllegalArgumentException e) {
            e.printStackTrace();
        } catch (IllegalAccessException e) {
            e.printStackTrace();
        } catch (InvocationTargetException e) {
            e.getTargetException().printStackTrace();
        }
    }


    public Collection<String> getMethodNames()
    {
        Method[] methods = this.getClass().getDeclaredMethods();
        ArrayList<String> names = new ArrayList<String>();
        for (int i = 0; i < methods.length; i++){
            String name = methods[i].getName();

            if (Modifier.isPublic(methods[i].getModifiers()) && 
                    methods[i].getParameterTypes().length == 1 && 
                    methods[i].getParameterTypes()[0] == double.class){
                names.add(name);
            }
        }
        Collections.sort(names);
        return names;
    }

    /*
     * TRAFFIC METRICS
     */


    private int getPMTraffic(double x){
        if (x > 340.0) {
            //trip was on PM1
            return pm1Traffic();
        } else {   
            return pm2Traffic();
        }
    }

    private int pm1Traffic(){
        if (tfa.getPM1Flow() < propsGlobal.getRequiredInt("traffic", "pm1", "low")) {
            return 0; //almost empty
        } else if (tfa.getPM1Flow() < propsGlobal.getRequiredInt("traffic", "pm1", "med")) {
            return 1; //low
        } else if (tfa.getPM1Flow() < propsGlobal.getRequiredInt("traffic", "pm1", "high")) {
            return 2; //med
        } else if (tfa.getPM1Flow() > propsGlobal.getRequiredInt("traffic", "pm1", "high")) {
            return 3; //high
        } else {
            return -1;
        }
    }


    private int pm2Traffic(){    	
        if (tfa.getPM2Flow() < propsGlobal.getRequiredInt("traffic", "pm2", "low")) {
            return 0; //almost empty
        } else if (tfa.getPM1Flow() < propsGlobal.getRequiredInt("traffic", "pm2", "med")) {
            return 1; //low
        } else if (tfa.getPM1Flow() < propsGlobal.getRequiredInt("traffic", "pm2", "high")) {
            return 2; //med
        } else if (tfa.getPM1Flow() > propsGlobal.getRequiredInt("traffic", "pm2", "high")) {
            return 3; //high
        } else {
            return -1;
        }
    }


    /*
     * LARGE EVENTS
     */

    public void none(double x){
        // do nothing.
    }

    public void thunderSparklePM1(double x) {
        logger.debug("thunderSparkle");

        Content sparkleThunderClip = anim.getContent("sparkleThunder");
        Clip faintSparkle = live.addClip(sparkleThunderClip, 320,0,canvas.getDimensions().width,16, 1.0, 10);
        faintSparkle.zIndex = -100; // sets to far background

        faintSparkle.delay(800).fadeOut(4500).delete();

        sc.playGlobal("EIA_organ_v01short.wav", false, 1.0f);
    }


    public void thunderSparklePM2big(double x) {
        logger.debug("thunderSparkle");

        Content sparkleThunderClip = anim.getContent("sparkleThunder");
        Clip faintSparkle = live.addClip(sparkleThunderClip, 0,0,320,16, 1.0, 10);
        faintSparkle.zIndex = -100; // sets to far background

        faintSparkle.delay(800).fadeOut(5000).delete();

        sc.playGlobal("EIA_organ_v01.wav", false, 1.0f);
        // or chime01_med.wav
    }


    public void thunderSparklePM2sm(double x) {
        logger.debug("thunderSparkle");

        Content sparkleThunderClip = anim.getContent("sparkleThunder");
        Clip faintSparkle = live.addClip(sparkleThunderClip, 0,0,320,16, 1.0, 10);
        faintSparkle.zIndex = -100; // sets to far background

        faintSparkle.delay(500).fadeOut(2500).delete();

        //sc.playGlobal("EIA_organ_v01short.wav", false, 1.0f);
        sc.playSingleChannelBlind("EIA_organ_v01short.wav", x, 1.0f);
    }


    public void bigHitPM1(double x){
        logger.debug("bigHit");

        int barWidth = 3;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip bigHit = live.addClip(simpleClip2,(int)x-barWidth/2,topBar,barWidth,barHeight*3, 0.1);

        Change out = new LinearChange().xTo(340).widthTo(300).alphaTo(0.4); // tuned to PM1
        Change hit = new LinearChange().alphaTo(1.0);

        bigHit.queueChange(out,820).queueChange(hit,10).fadeOut(1000).delete();

        sc.playSingleChannelBlind("66_Reverse_Orchestra_Hit.wav", x, 1.0f);
    }


    public void grow(double x){
        logger.debug("grow");

        int barWidth = 3;
        int dist = 65;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip grow = live.addClip(simpleClip2,(int)x-barWidth,topBar,barWidth,barHeight*3, 1.0);

        Change out = new LinearChange().xTo(x-dist).widthTo(dist);
        Change back = new LinearChange().xTo(x+lookAhead*1).widthTo(barWidth);

        grow.queueChange(out,600).queueChange(back,600).fadeOut(100).delete();
        
        Set<String> set = new LinkedHashSet<String>(3);
        set.add("highlight_22.wav");
        set.add("highlight_14.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }  
    }
    
    public void boomerang(double x){
        logger.debug("boomerang");

        int barWidth = 18;
        int dist = 65;

        x = findNearestLight(x+lookAhead/2,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip boomerang = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight*3, 1.0);

        Change out = new LinearChange().xTo(x-dist).scaleWidth(1.5);
        //Change back = new LinearChange().xTo(x+lookAhead*2).scaleWidth(0.66).alphaTo(0.1);
        Change back = new LinearChange().xTo(x+lookAhead*2).scaleWidth(0.66);

        boomerang.queueChange(out, 500).queueChange(back,500).fadeOut(100).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("highlight_13.wav");
        set.add("highlight_06.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        } 

    }


    public void entryWavePM1(double x) {
        logger.debug("entryWavePM1");
        x = findNearestLight(x+lookAhead,true);

        Content waveImage = anim.getContent("waveImage");

        int waveWidth = 64;
        Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,cHeight, 1.0); //add it as 32px wide at the end of the stage
        waveImageClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(320);
        //waveImageClip.queueChange(waveMove, 1500).delete();
        waveImageClip.queueChange(waveMove, 2300).delete(); //longer due to diff length ppm

        int kotuNum = (int)(Math.random()*5+1);
        String kotuFile = "kotu_0"+kotuNum+".wav";
        //logger.info("EIAClipPlayer - play kotuFile: " + kotuFile);
        sc.playSingleChannelBlind(kotuFile, x, 1.0f);
    }

    public void entryWavePM2(double x) {
        logger.debug("entryWavePM2");
        x = findNearestLight(x+lookAhead,true);

        Content waveImage = anim.getContent("waveImage");

        int waveWidth = 64;
        Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,cHeight, 1.0); //add it as 32px wide at the end of the stage
        waveImageClip.zIndex = -100; // sets to far background

        //fadein, wait, fadeout
        Change waveMove = new LinearChange().xTo(-waveWidth);
        //waveImageClip.queueChange(waveMove, 1500).delete();
        waveImageClip.queueChange(waveMove, 2100).delete();

        int kotuNum = (int)(Math.random()*5+1);
        String kotuFile = "kotu_0"+kotuNum+".wav";
        //logger.info("EIAClipPlayer - play kotuFile: " + kotuFile);
        sc.playSingleChannelBlind(kotuFile, x, 1.0f);
    }

    public void bigVertDoublet(double x) {
        logger.debug("bigVertDoublet");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 32;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth+2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 1.0, 1000); 

        topBlip.delay(600).fadeOut(600).delete();
        bottomBlip.delay(800).fadeOut(600).delete();

        // something bigger here
        sc.playSingleChannelBlind("piano_doublet01.wav", x, 1.0f);

    }

    public void bigVertDoubletSparkle(double x) {
        logger.debug("bigVertDoublet");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 32;

        Content doubletSparkle = anim.getContent("sparkleDoublet");
        Content doubletSparkle2 = anim.getContent("sparkleDoublet");
        //Clip faintSparkle = live.addClip(sparkleThunderClip, 320,0,canvas.getDimensions().width,16, 1.0, 10);

        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(doubletSparkle, (int)x-barWidth+2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(doubletSparkle2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 1.0, 1000);
        Clip topBlipBg = live.addClip(simpleClip2, (int)x-barWidth+2,topBar,barWidth,barHeight, 0.3); 
        Clip bottomBlipBg = live.addClip(simpleClip2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 0.3, 1000); 

        topBlip.delay(600).fadeOut(600).delete();
        bottomBlip.delay(800).fadeOut(600).delete();
        topBlipBg.delay(600).fadeOut(600).delete();
        bottomBlipBg.delay(800).fadeOut(600).delete();

        // something bigger here
        sc.playSingleChannelBlind("piano_doublet01.wav", x, 1.0f);

    }

    public void bigVertDoubletRandom(double x) {
        logger.debug("bigVertDoublet2");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 32;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Content doubletSparkle = anim.getContent("sparkleDoublet");
        Content doubletSparkle2 = anim.getContent("sparkleDoublet");

        Clip topBlip = live.addClip(doubletSparkle, (int)x-barWidth+2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(doubletSparkle2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 1.0, 1000);
        Clip topBlipBg = live.addClip(simpleClip2, (int)x-barWidth+2,topBar,barWidth,barHeight, 0.4); 
        Clip bottomBlipBg = live.addClip(simpleClip2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 0.4, 1000); 

        topBlip.delay(600).fadeOut(600).delete();
        bottomBlip.delay(800).fadeOut(600).delete();
        topBlipBg.delay(600).fadeOut(600).delete();
        bottomBlipBg.delay(800).fadeOut(600).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("EchoPiano1.wav");
        set.add("EchoPiano2.wav");
        set.add("EchoPiano3.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                //logger.info("Random String = " + file);
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   

    }




    public void randomBars(double x){
        if (getPMTraffic(x) > 1){
            randomBarsLess(x);
        } else {
            randomBarsMore(x);
        }
    }

    public void randomBarsLess(double x){
        logger.debug("randomBars");
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

        sc.playSingleChannelBlind("vert_connect_med_whoosh16.wav", x, 1.0f);
    }

    public void randomBarsMore(double x){
        logger.debug("randomBarsMore");
        int maxLoop = 24;
        int minLoop = 14;
        int loop = (int)(maxLoop*Math.random());
        int delay = 120;
        boolean top = true;
        for (int l=0; l<Math.max(loop,minLoop); l++){
            if (top){
                topRandomBar(x,delay*l,7,-50);
                top = false;
            } else {
                bottomRandomBar(x,delay*l,7,-50);
                top = true;
            }
        }

        //vert_connect_med_whoosh16.wav
        sc.playSingleChannelBlind("vert_connect_med_whoosh16_long.wav", x, 1.0f);
    }

    private int randBarSpeed = 1500;

    public void topRandomBar(double x, int delay, int maxBarLength, int barDest){
        //logger.debug("topRandomBar");

        int barWidth = (int)(maxBarLength*Math.random())+1;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight, 1.0, delay);

        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(250).delete();
    }

    public void bottomRandomBar(double x, int delay, int maxBarLength, int barDest){
        //logger.debug("bottomRandomBar");

        int barWidth = (int)(maxBarLength*Math.random())+1;

        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,bottomBar,barWidth,barHeight, 1.0, delay);

        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(250).delete();
    }





    /*
     * V1 Wave Animations
     * 

     */

    public void waveBlockAll(double x) {
        //logger.info("blockWaveAll@ " + x);
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

    public void waveBigAll(double x) {
        //logger.info("bigWaveAll@ " + x);
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

    public void waveMegaDouble(double x) {
        //logger.info("megaWaveDouble@ " + x);
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

        wave1.queueChange(waveMove1, 3000).delete();
        wave2.queueChange(waveMove2, 3000).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("BondiSwell_a.wav");
        set.add("BondiSwell_b.wav");
        set.add("BondiSwell_c.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                //logger.info("Random String = " + file);
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   
    }
    
    public void waveMegaDoubleAnalog(double x) {
        //logger.info("megaWaveDouble@ " + x);
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

        wave1.queueChange(waveMove1, 3000).delete();
        wave2.queueChange(waveMove2, 3000).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("AnalogSwell_a.wav");
        set.add("AnalogSwell_b.wav");
        set.add("AnalogSwell_c.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                //logger.info("Random String = " + file);
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   
    }

    public void waveMega(double x) {
        //logger.info("megaWaveDouble@ " + x);
        Content waveImage = anim.getContent("megaWaveFull");

        //int waveWidth = 16;
        int waveWidth = 16;
        Clip wave1 = live.addClip(waveImage, (int)(x-waveWidth/2),0,waveWidth,cHeight, 1.0); 
        //wave1.zIndex = -100; // sets to far background

        //Change waveMove1 = new LinearChange().xTo(wave1End-waveWidth).scaleWidth(3.0).alphaTo(0.0);
        //double newScale = 12.0;
        double newScale = 20.0;
        // first start a scale operation
        Change waveScale1 = new LinearChange().xTo(x-waveWidth/2*newScale/2).scaleWidth(newScale/2);
        Change waveScale2Fade = new LinearChange().xTo(x-waveWidth/2*newScale).scaleWidth(newScale/2).alphaTo(0.0);

        //wave1.queueChange(waveMove1, 2200).delete();
        wave1.queueChange(waveScale1, 1200).queueChange(waveScale2Fade, 1200).delete();
        //wave1.queueChange(waveScale2Fade, 1200).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("BondiSwell_a.wav");
        set.add("BondiSwell_b.wav");
        set.add("BondiSwell_c.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                //logger.info("Random String = " + file);
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }  
    }
    
    public void waveMegaAnalog(double x) {
        Content waveImage = anim.getContent("megaWaveFull");

        //int waveWidth = 16;
        int waveWidth = 16;
        Clip wave1 = live.addClip(waveImage, (int)(x-waveWidth/2),0,waveWidth,cHeight, 1.0); 
        //wave1.zIndex = -100; // sets to far background

        //Change waveMove1 = new LinearChange().xTo(wave1End-waveWidth).scaleWidth(3.0).alphaTo(0.0);
        //double newScale = 12.0;
        double newScale = 20.0;
        // first start a scale operation
        Change waveScale1 = new LinearChange().xTo(x-waveWidth/2*newScale/2).scaleWidth(newScale/2);
        Change waveScale2Fade = new LinearChange().xTo(x-waveWidth/2*newScale).scaleWidth(newScale/2).alphaTo(0.0);

        //wave1.queueChange(waveMove1, 2200).delete();
        wave1.queueChange(waveScale1, 1200).queueChange(waveScale2Fade, 1200).delete();
        //wave1.queueChange(waveScale2Fade, 1200).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("AnalogSwell_a.wav");
        set.add("AnalogSwell_b.wav");
        set.add("AnalogSwell_c.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                //logger.info("Random String = " + file);
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }  
    }


    public void wavesRandom(double x) {
        //logger.info("randomWaves @" + x);
        int xLow = 0;
        int xHigh = cWidth;
        int waveWidth = 32;

        int waves = 10;
        int pDelay = eNote;
        Content waveImage = anim.getContent("waveImage");
        for (int p=0; p<waves; p++){
            int newx = (int)(Math.random()*(xHigh-xLow));
            Change moveIt = new LinearChange().xBy(-16).alphaTo(0.0);
            live.addClip(waveImage, newx, 0, waveWidth, cHeight, 1.0, p*pDelay).queueChange(moveIt, 1500);
        }

        sc.playSingleChannelBlind("BondiSwell_b.wav", x, 1.0f);
    }


    public void wavesRandom2(double x) {
        //logger.info("randomWaves2 @" + x);
        int xLow = 0;
        int xHigh = cWidth;
        int waveWidth = 40;

        int waves = 10;
        int pDelay = eNote/3;
        Content waveImage = anim.getContent("waveImage");
        for (int p=0; p<waves; p++){
            int newx = (int)(Math.random()*(xHigh-xLow));
            Change moveIt = new LinearChange().xBy(-16).alphaTo(0.0);
            live.addClip(waveImage, newx, 0, waveWidth, cHeight, 1.0, p*pDelay).queueChange(moveIt, 1500);
        }

        sc.playSingleChannelBlind("AnalogSwell_a.wav", x, 1.0f);
    }

    public void wavesStructured(double x) {
        //logger.info("structuredWaves @" + x);
        int waves = 10; //?
        int wave1End = 340;
        int waveWidth = 32;
        int waveOffset = waveWidth/3;
        int waveSpacing = waveWidth + waveOffset;
        waves = (cWidth-wave1End)/waveSpacing;

        int pDelay = eNote;
        Content waveImage = anim.getContent("waveImage");
        for (int p=0; p<waves; p++){
            int newx1 = wave1End + p*waveSpacing; //ppm 1
            int newx2 = 0 + p*waveSpacing; //ppm 2
            Change moveIt = new LinearChange().xBy(-1*waveWidth/2).alphaTo(0.0);
            live.addClip(waveImage, newx1, 0, waveWidth, cHeight, 1.0, p*pDelay).queueChange(moveIt, 1500);
            live.addClip(waveImage, newx2, 0, waveWidth, cHeight, 1.0, p*pDelay).queueChange(moveIt, 1500);
        }

        sc.playSingleChannelBlind("AnalogSwell_b.wav", x, 1.0f);
    }



    /*
     * ACCENTS
     * 
     * 
     */

    public void accentSm(double x) {
        int rand = (int)(Math.random() * 100);

        if (rand < 45){
            if (getPMTraffic(x) == 0) {
                accentVertDoubletFour(x);
            } else if (getPMTraffic(x) < 1) {
                accentVertDoubletThree(x);
            } else {
                accentVertDoubletTwo(x);
            }
        } else if (rand < 100) {
            if (getPMTraffic(x) < 2) {
                accentVertSixFill(x);
            } else {
                accentVertFourFill(x);
            }
        } else {
            accentBlipMarimba(x); // default, not used
        }

    }
    
    public void accentSmSplit(double x) {

        if (x > 340){
            if (getPMTraffic(x) == 0) {
                accentVertDoubletFour(x);
            } else if (getPMTraffic(x) < 1) {
                accentVertDoubletThree(x);
            } else {
                accentVertDoubletTwo(x);
            }
        } else {
            if (getPMTraffic(x) < 2) {
                accentVertSixFill(x);
            } else {
                accentVertFourFill(x);
            }
        }

    }

    public void accentBlipStorm(double x){
        int count = 0;
        if (getPMTraffic(x) >= 2) { //play upright bass
            count = 5;
            for (int i=0; i< count; i++){
                Timer blipTimer = new Timer();
                double offset = Math.random() * 30 - 15;
                UprightBassStormTask mt = new UprightBassStormTask();
                mt.setX(x + offset);
                blipTimer.schedule(mt, 220*i);
            }
        } else if (getPMTraffic(x) < 2) {
            count = 15;
            int range =  60;
            double rand = Math.random()*100;
            for (int i=0; i< count; i++){
                Timer blipTimer = new Timer();
                double offset = Math.random() * range - range/2;
                if (rand < 50) {
                    MarimbaStormTask mt = new MarimbaStormTask();
                    mt.setX(x + offset);
                    blipTimer.schedule(mt, 125*i);
                } else if (rand > 50){
                    VibraphoneStormTask mt = new VibraphoneStormTask();
                    mt.setX(x + offset);
                    blipTimer.schedule(mt, 125*i);
                }
            }
        }
    }
    
    public void accentBlipStormScaled(double x){
        int count = 0;
        int range =  0;
        if (getPMTraffic(x) >= 2) { //play upright bass
            count = 9;
            range =  40;
        } else if (getPMTraffic(x) < 2) {
            count = 16;
            range =  60;
        }
        double rand = Math.random()*100;
        for (int i=0; i< count; i++){
            Timer blipTimer = new Timer();
            double offset = Math.random() * range - range/2;
            if (rand < 50) {
                MarimbaStormTask mt = new MarimbaStormTask();
                mt.setX(x + offset);
                blipTimer.schedule(mt, 125*i);
            } else if (rand > 50){
                VibraphoneStormTask mt = new VibraphoneStormTask();
                mt.setX(x + offset);
                blipTimer.schedule(mt, 125*i);
            }
        }
    }
    
    //TEST!!!!
    public void accentBlipWalkScaled(double x){
        int count = 0;
        int range =  0;
        double barWidth = 3.75;
        if (getPMTraffic(x) >= 2) { //play upright bass
            count = 9;
            range =  40;
        } else if (getPMTraffic(x) < 2) {
            count = 16;
            range =  60;
        }
        double rand = Math.random()*100;
        for (int i=0; i< count; i++){
            Timer blipTimer = new Timer();
            double offset = i * barWidth - (count*barWidth)/2;
            if (rand < 50) {
                MarimbaStormTask mt = new MarimbaStormTask();
                mt.setX(x + offset);
                blipTimer.schedule(mt, 125*i);
            } else if (rand > 50){
                VibraphoneStormTask mt = new VibraphoneStormTask();
                mt.setX(x + offset);
                blipTimer.schedule(mt, 125*i);
            }
        }
    }

    public void accentUprightBassStorm(double x){
        int count = 6;
        int range = 26;
        for (int i=0; i< count; i++){
            Timer blipTimer = new Timer();
            double offset = Math.random() * range - range/2;
            UprightBassStormTask mt = new UprightBassStormTask();
            mt.setX(x + offset);
            blipTimer.schedule(mt, 220*i);
        }
    }
    
    public void accentUprightBassWalk(double x){
        int count = 6;
        int range = 26;
        int barWidth = 4;
        for (int i=0; i< count; i++){
            Timer blipTimer = new Timer();
            double offset = i * barWidth - (count*barWidth)/2;
            UprightBassStormTask mt = new UprightBassStormTask();
            mt.setX(x + offset);
            blipTimer.schedule(mt, 220*i);
        }
    }

    private class MarimbaStormTask extends TimerTask {
        double x;
        public void setX(double x){
            this.x = x;
        }
        public void run() {
            accentBlipMarimba(x);
        }
    }

    private class VibraphoneStormTask extends TimerTask {
        double x;
        public void setX(double x){
            this.x = x;
        }
        public void run() {
            accentBlipVibraphone(x);
        }
    }

    private class UprightBassStormTask extends TimerTask {
        double x;
        public void setX(double x){
            this.x = x;
        }
        public void run() {
            accentBlipUprightBass(x);
        }
    }

    public void accentBlipUprightBass(double x) {
        logger.debug("bass blip");
        x = findNearestLight(x+lookAhead,true);
        //orig int barWidth = 3;
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0); //set the alpha to 0.5 to get 50% brightness on creation
        blip1.delay(100).fadeOut(500).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("UprightJazzBassBlues_a.wav");
        set.add("UprightJazzBassBlues_b.wav");
        set.add("UprightJazzBassBlues_c.wav");
        set.add("UprightJazzBassBlues_d.wav");
        set.add("UprightJazzBassBlues_e.wav");
        set.add("UprightJazzBassBlues_f.wav");
        set.add("UprightJazzBassBlues_g.wav");

        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   
    }

    public void accentBlipMarimba(double x) {
        logger.debug("marimba blip");
        x = findNearestLight(x+lookAhead,true);
        //orig int barWidth = 3;
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0); //set the alpha to 0.5 to get 50% brightness on creation
        blip1.delay(100).fadeOut(500).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("marimba_mid_01a.wav");
        set.add("marimba_mid_01b.wav");
        set.add("marimba_mid_02a.wav");
        set.add("marimba_mid_02b.wav");
        set.add("marimba_mid_03a.wav");
        set.add("marimba_mid_04b.wav");
        set.add("marimba_mid_04a.wav");
        set.add("marimba_mid_04b.wav");
        set.add("marimba_mid_05a.wav");
        set.add("marimba_mid_05b.wav");
        set.add("marimba_mid_06a.wav");
        set.add("marimba_mid_06b.wav");
        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   
    }

    public void accentBlipVibraphone(double x) {
        logger.debug("bass blip");
        x = findNearestLight(x+lookAhead,true);
        //orig int barWidth = 3;
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 1.0); //set the alpha to 0.5 to get 50% brightness on creation
        blip1.delay(100).fadeOut(500).delete();

        Set<String> set = new LinkedHashSet<String>(3);
        set.add("CarribeanVibraphoneBlues_a.wav");
        set.add("CarribeanVibraphoneBlues_b.wav");
        set.add("CarribeanVibraphoneBlues_c.wav");
        set.add("CarribeanVibraphoneBlues_d.wav");
        set.add("CarribeanVibraphoneBlues_e.wav");
        set.add("CarribeanVibraphoneBlues_f.wav");
        set.add("CarribeanVibraphoneBlues_g.wav");

        int size = set.size();
        int item = new Random().nextInt(size);
        int i = 0;
        for(Object file : set)
        {
            if (i == item){
                sc.playSingleChannelBlind((String)file, x, 1.0f);
            }
            i++;
        }   
    }


    public void accentStrobeRand(double x) {
        logger.debug("strober");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int dOff = 150; //delay offset

        int barHLoc = 0;
        if (Math.random() > 0.5) {
            barHLoc = topBar;
        } else {
            barHLoc = bottomBar;
        }


        Clip top1 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*0),barHLoc,barWidth,barHeight, 1.0); 
        Clip top2 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*1),barHLoc,barWidth,barHeight, 1.0, dOff*1); 
        Clip top3 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*2),barHLoc,barWidth,barHeight, 1.0, dOff*2);
        Clip top4 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*3),barHLoc,barWidth,barHeight, 1.0, dOff*3); 
        Clip top5 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*4),barHLoc,barWidth,barHeight, 1.0, dOff*4); 
        Clip top6 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*5),barHLoc,barWidth,barHeight, 1.0, dOff*5); 
        Clip top7 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*6),barHLoc,barWidth,barHeight, 1.0, dOff*6); 
        Clip top8 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*7),barHLoc,barWidth,barHeight, 1.0, dOff*7); 

        int delay = 350;
        int fadetime = 350;
        top1.delay(delay).fadeOut(fadetime).delete();
        top2.delay(delay).fadeOut(fadetime).delete();
        top3.delay(delay).fadeOut(fadetime).delete();
        top4.delay(delay).fadeOut(fadetime).delete();
        top5.delay(delay).fadeOut(fadetime).delete();
        top6.delay(delay).fadeOut(fadetime).delete();
        top7.delay(delay).fadeOut(fadetime).delete();
        top8.delay(delay).fadeOut(fadetime).delete();


        //sc.playSingleChannelBlind("entrance6.wav", x, 0.5f);
    }


    public void accentVertSixFill(double x) {
        logger.debug("vertSixFill");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int dOff = 150; //delay offset

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

        //sc.playSingleChannelBlind("entrance6.wav", x, 0.5f);
        sc.playSingleChannelBlind("entrance6.wav", x, 1.0f);
    }

    public void accentVertFourFill(double x) {
        logger.debug("vertSixFill");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        int dOff = 150; //delay offset

        Clip top1 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*0),topBar,barWidth,barHeight, 1.0); 
        Clip bottom1 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*0),bottomBar,barWidth,barHeight, 1.0, dOff*1);
        Clip top2 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*1),topBar,barWidth,barHeight, 1.0, dOff*2); 
        Clip bottom2 = live.addClip(simpleClip2, (int)(x-barWidth/2+barOff*1),bottomBar,barWidth,barHeight, 1.0, dOff*3); 

        top1.delay(350).fadeOut(350).delete();
        bottom1.delay(350).fadeOut(350).delete();
        top2.delay(350).fadeOut(350).delete();
        bottom2.delay(350).fadeOut(350).delete();

        //sc.playSingleChannelBlind("entrance6_short.wav", x, 0.5f);
        sc.playSingleChannelBlind("entrance6_short.wav", x, 1.0f);
    }


    public void accentVertDoubletFour(double x) {
        logger.debug("smVertDoublet");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 13;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2-2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2-2,bottomBar,barWidth,barHeight, 1.0, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.5f);
        sc.playSingleChannelBlind("lumen_entrance7.wav", x, 1.0f);
    }

    public void accentVertDoubletThree(double x) {
        logger.debug("smVertDoublet");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 9;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2,bottomBar,barWidth,barHeight, 1.0, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.5f);
        sc.playSingleChannelBlind("lumen_entrance7.wav", x, 1.0f);
    }

    public void accentVertDoubletTwo(double x) {
        logger.debug("smVertDoublet");
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 6;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);

        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2-1,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2-1,bottomBar,barWidth,barHeight, 1.0, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.5f);
        sc.playSingleChannelBlind("lumen_entrance7.wav", x, 1.0f);
    }


    public void testClip(double x) {
        logger.info("ClipPlayer - testClip starting, playing accentVertDoubletTwo()");
        accentVertDoubletTwo(x);
    }
    
    public void testSensor(double x) {
        logger.debug("sensor test blip");
        x = findNearestLight(x,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 0.5); //set the alpha to 0.5 to get 50% brightness on creation
        blip1.delay(1000).fadeOut(500).delete();
    }


    /**
     * Local Util Methods
     */
    public double findNearestLight(double x, boolean forward) {

        double closestX = -20;

        for (Fixture f: elu.getFixtures()) {
            if (Math.abs(x-f.getLocation().x) < Math.abs(x-closestX)) {
                closestX = f.getLocation().x;
            }
        }
        ////logger.info("ClipPlayer: Track x= " + x + " & closest fixture x= " + closestX);
        return closestX;
    }
}