package net.electroland.edmonton.core;

import java.awt.Color;
import java.lang.reflect.InvocationTargetException;
import java.lang.reflect.Method;
import java.lang.reflect.Modifier;
import java.util.ArrayList;
import java.util.Collection;
import java.util.Collections;
import java.util.Map;
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
		
		phantomTimer = new Timer();
		phantomTimer.schedule(new phantomTimerTask(), 1000, 1000);
		live = anim.addClip(new SolidColorContent(null), 0, 0, anim.getStageDimensions().width, anim.getStageDimensions().height, 1.0);
		logger.info("ClipPlayer2 created: " + this);
	}
	
	private class phantomTimerTask extends TimerTask {
		public void run() {
			// fire phantom
			smVertDoublet(Math.random() * 620);			
			logger.info(this + " fired a phantom clip");
		}
	}

	public void playClip(String name, double loc){
		
		logger.info("I AM ClipPlayer2: " + this);

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

	private double barOff = -3.69;
	private double lookAhead = -3.2;
	private int topBar = 5;
	private int bottomBar = 8;
	private int barHeight = 2;

	public void none(double x){
		// do nothing.
	}
	/*
	 * BIG Animations
	 */

	public void thunderSparklePM1(double x) {
		logger.debug("thunderSparkle");

		Content sparkleThunderClip = anim.getContent("sparkleThunder");
		Clip faintSparkle = live.addClip(sparkleThunderClip, 320,0,canvas.getDimensions().width,16, 1.0, 10);
		faintSparkle.zIndex = -100; // sets to far background

		faintSparkle.delay(800).fadeOut(4500).delete();

		sc.playGlobal("EIA_organ_v01short.wav", false, 1.0f);
	}


	public void thunderSparklePM2(double x) {
		if (pm2Traffic() < 1){
			thunderSparklePM2big(x);
		} else {
			thunderSparklePM2sm(x);
		}
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
		sc.playSingleChannelBlind("highlight_22.wav", x, 1.0f);
	}


	public void entryWavePM1(double x) {
		logger.debug("entryWavePM1");
		x = findNearestLight(x+lookAhead,true);

		Content waveImage = anim.getContent("waveImage");

		int waveWidth = 48;
		Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
		waveImageClip.zIndex = -100; // sets to far background

		//fadein, wait, fadeout
		Change waveMove = new LinearChange().xTo(320);
		waveImageClip.queueChange(waveMove, 1500).delete();

		int kotuNum = (int)(Math.random()*5+1);
		String kotuFile = "kotu_0"+kotuNum+".wav";
		//logger.info("EIAClipPlayer - play kotuFile: " + kotuFile);
		sc.playSingleChannelBlind(kotuFile, x, 1.0f);
	}

	public void entryWavePM2(double x) {
		logger.debug("entryWavePM2");
		x = findNearestLight(x+lookAhead,true);

		Content waveImage = anim.getContent("waveImage");

		int waveWidth = 48;
		Clip waveImageClip = live.addClip(waveImage, (int)x,0,waveWidth,16, 1.0); //add it as 32px wide at the end of the stage
		waveImageClip.zIndex = -100; // sets to far background

		//fadein, wait, fadeout
		Change waveMove = new LinearChange().xTo(-waveWidth);
		waveImageClip.queueChange(waveMove, 1500).delete();

		sc.playSingleChannelBlind("kotu_04.wav", x, 1.0f);
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

		sc.playSingleChannelBlind("vert_disconnect_long_whoosh03.wav", x, 1.0f);
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

	public void randomBars(double x){
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

		sc.playSingleChannelBlind("vert_connect_med_whoosh16_long.wav", x, 1.0f);
	}

	public void randomBarsMore(double x){
		logger.debug("randomBarsMore");
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
	 * SMALL Animations
	 */

	public void smAccent(double x) {
		int rand = (int)(Math.random() * 100);
		//logger.debug(rand);
		if (rand < 40){
			smVertDoublet(x);
		} else if (rand < 100) {
			vertSixFill(x);
		} else {
			blip2(x); // default, not used
		}

	}


	public void blip2(double x) {
		logger.debug("blip2");
		x = findNearestLight(x+lookAhead,true);
		//orig int barWidth = 3;
		int barWidth = 3;
		Content simpleClip2 = new SolidColorContent(Color.WHITE);
		Clip blip1 = live.addClip(simpleClip2, (int)x-barWidth/2,0,barWidth,16, 0.8); //set the alpha to 0.5 to get 50% brightness on creation
		blip1.delay(100).fadeOut(400).delete();

		int rand = (int)(Math.random() * 100);
		if (rand < 50){
			sc.playSingleChannelBlind("marimba_mid_01a.wav", x, 0.5f); 
		} else {
			sc.playSingleChannelBlind("marimba_mid_01b.wav", x, 0.5f); 
		} 
	}


	// maybe do a faster version?
	public void vertSixFill(double x) {
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

		sc.playSingleChannelBlind("entrance6.wav", x, 0.5f);
	}


	public void smVertDoublet(double x) {
		logger.debug("smVertDoublet");
		x = findNearestLight(x+lookAhead,true);
		int barWidth = 9;
		Content simpleClip2 = new SolidColorContent(Color.WHITE);

		Clip topBlip = live.addClip(simpleClip2, (int)x-barWidth/2,topBar,barWidth,barHeight, 1.0); 
		Clip bottomBlip = live.addClip(simpleClip2, (int)x-barWidth/2,bottomBar,barWidth,barHeight, 1.0, 250); 

		topBlip.delay(500).fadeOut(500).delete();
		bottomBlip.delay(500).fadeOut(500).delete();

		sc.playSingleChannelBlind("lumen_entrance7.wav", x, 0.5f);
	}


	public void testClip(double x) {
		logger.info("ClipPlayer - testClip starting");
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
		logger.info("ClipPlayer - testClip animation done, playing sound");

		sc.playSingleChannelBlind("entrance6.wav", x, 0.5f);
		logger.info("ClipPlayer - testClip sound call made");
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