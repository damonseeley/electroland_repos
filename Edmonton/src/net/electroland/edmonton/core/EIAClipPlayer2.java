package net.electroland.edmonton.core;

import java.awt.Color;

import net.electroland.ea.AnimationManager;
import net.electroland.ea.Clip;
import net.electroland.ea.Content;
import net.electroland.ea.content.SolidColorContent;
import net.electroland.utils.lighting.ELUManager;

import org.apache.log4j.Logger;

public class EIAClipPlayer2 extends EIAClipPlayer {

    static Logger logger = Logger.getLogger(EIAClipPlayer2.class);

    public EIAClipPlayer2(AnimationManager am, ELUManager elu, SoundController sc)
    {
        super(am, elu, sc);
    }

    public void test(double x){
        //logger.info("we're on at " + x);
        blipVertDoublet(x);
    }

    private double barOff = -3.69;
    private double lookAhead = -5.1;
    //private double blipLookAhead = -6.2;
    private double blipLookAhead = -3.2;

    public void blip2(double x) {

        boolean doBlip = true;
        boolean doBlipSound = true;

        if (doBlip){
            x = findNearestLight(x+blipLookAhead,true);
            //orig int barWidth = 3;
            int barWidth = 3;
            Content simpleClip2 = new SolidColorContent(Color.WHITE);
            Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 0.7); //set the alpha to 0.5 to get 50% brightness on creation
            blip1.delay(150).fadeOut(400).delete();

            //now play a sound!
            if (doBlipSound) {
                //sc.playSingleChannelBlind("marimba_mid_01.wav", x, 0.5f); // the firs test, kind of "crunch" sound
            }
        }
    }
    
    private int topBar = 6;
    private int bottomBar = 9;
    private int barHeight = 2;

    public void blipVertDoublet(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+blipLookAhead,true);
        int barWidth = 3;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2,topBar,barWidth,barHeight, 0.9); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2,bottomBar,barWidth,barHeight, 0.9, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //now play a sound!
        //sc.playSingleChannelBlind("lumen_3.wav", x, 0.4f); // the firs test, kind of "crunch" sound

    }

    
    
    
    

}