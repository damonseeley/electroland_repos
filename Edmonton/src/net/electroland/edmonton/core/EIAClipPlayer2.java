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

    public void blip2(double x) {

        boolean doBlip = true;
        boolean doBlipSound = true;

        if (doBlip){
            x = findNearestLight(x+lookAhead,true);
            //orig int barWidth = 3;
            int barWidth = 3;
            Content simpleClip2 = new SolidColorContent(Color.WHITE);
            Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 0.7); //set the alpha to 0.5 to get 50% brightness on creation
            blip1.delay(150).fadeOut(400).delete();

            //now play a sound
            if (doBlipSound) {
                //sc.playSingleChannelBlind("marimba_mid_01.wav", x, 0.5f); // the firs test, kind of "crunch" sound
            }
        }
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
	}
    
    public void boomerang(double x){
    	logger.info("boomerang");
    	
    	int barWidth = 5;
    	int dist = 65;
    	
        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip boom = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight*3, 1.0);
        
        Change out = new LinearChange().xTo(x-dist).scaleWidth(5.0);
        Change back = new LinearChange().xTo(x+lookAhead*2).scaleWidth(0.2);
        
        boom.queueChange(out, 800).queueChange(back,800).fadeOut(300).delete();
    }
    
    public void randomBars(double x){
    	int maxLoop = 13;
    	int minLoop = 7;
    	int loop = (int)(maxLoop*Math.random());
    	int delay = 250;
    	boolean top = true;
    	for (int l=0; l<Math.max(loop,minLoop); l++){
    		if (top){
    			topRandomBar(x,delay*l);
    			top = false;
    		} else {
    			bottomRandomBar(x,delay*l);
    			top = true;
    		}
    	}
    }
    
    private int randBarSpeed = 1500;
    
    public void topRandomBar(double x, int delay){
    	
    	logger.info("topRandomBar");
    	
    	int maxBarLength = 6;
    	int barWidth = (int)(maxBarLength*Math.random())+1;
    	int barDest = -32;
    	
        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,topBar,barWidth,barHeight, 1.0, delay);
        
        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(500).delete();
    }
    
    public void bottomRandomBar(double x, int delay){
    	
    	logger.info("bottomRandomBar");
    	
    	int maxBarLength = 7;
    	int barWidth = (int)(maxBarLength*Math.random())+1;
    	int barDest = -32;
    	
        x = findNearestLight(x+lookAhead,true);
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip bar = live.addClip(simpleClip2, (int)x-barWidth,bottomBar,barWidth,barHeight, 1.0, delay);
        
        Change barMove = new LinearChange().xTo(x+barDest);
        bar.queueChange(barMove, randBarSpeed).fadeOut(500).delete();
    }



    public void blipVertDoublet(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 9;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2,bottomBar,barWidth,barHeight, 1.0, 250); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //now play a sound!
        sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.4f);

    }
    
    public void blipBigVertDoublet(double x) {

        //logger.info("localStabSmall@ " + x);
        x = findNearestLight(x+lookAhead,true);
        int barWidth = 22;
        Content simpleClip2 = new SolidColorContent(Color.WHITE);
        
        Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth+2,topBar,barWidth,barHeight, 1.0); 
        Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth+2,bottomBar,barWidth,barHeight, 1.0, 300); 

        topBlip.delay(500).fadeOut(500).delete();
        bottomBlip.delay(500).fadeOut(500).delete();

        //now play a sound!
        // something bigger here
        //sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.4f);

    }
    


    
    
    
    

}