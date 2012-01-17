package net.electroland.edmonton.core;

import java.awt.Rectangle;

import org.apache.log4j.Logger;

import net.electroland.ea.AnimationManager;
import net.electroland.edmonton.core.sequencing.ClipCue;
import net.electroland.utils.lighting.ELUManager;
import net.electroland.utils.lighting.Fixture;

public class EIAClipPlayer {

	private AnimationManager anim;
	private ELUManager elu;

	static Logger logger = Logger.getLogger(EIAClipPlayer.class);

	public EIAClipPlayer(AnimationManager am, ELUManager elu)
	{
		this.anim = am;
		this.elu = elu;
	}


	/*
	 * EIA Show methods
	 */

	public void localTrill4Up(double x) {
		//logger.info("localTrill4Up@ " + x);

		x = findNearestLight(x,true);
		//8 px wide
		int barWidth = 3;
		int of = 1; //offset to hit lights

		//create all bars, but at 0.0 alpha to popin later
		int trill1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth*2-of,0,barWidth,16), 0.0); //2px wide rect at left of area
		int trill2 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth-of,0,barWidth,16), 0.0);
		int trill3 = anim.startClip("simpleClip2", new Rectangle((int)x-of,0,barWidth,16), 0.0); 
		int trill4 = anim.startClip("simpleClip2", new Rectangle((int)x+barWidth-of,0,barWidth,16), 0.0); 

		/*
		int trill1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth*2,0,barWidth,16), 0.0); //2px wide rect at left of area
		int trill2 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth,0,barWidth,16), 0.0);
		int trill3 = anim.startClip("simpleClip2", new Rectangle((int)x,0,barWidth,16), 0.0); 
		int trill4 = anim.startClip("simpleClip2", new Rectangle((int)x+barWidth,0,barWidth,16), 0.0); 
		 */

		//now pop in trills2-4 on time - 60,230,435,590 abs time, shift 60 for local
		anim.queueClipChange(trill1, null, null, 1.0, 0, 0, false); //popin 
		anim.queueClipChange(trill2, null, null, 1.0, 0, 170, false); //popin 
		anim.queueClipChange(trill3, null, null, 1.0, 0, 375, false); //popin 
		anim.queueClipChange(trill4, null, null, 1.0, 0, 530, false); //popin 

		//fade em all out
		anim.queueClipChange(trill1, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill2, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill3, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill4, null, null, 0.0, 500, 3500, true); //fadeout
	}


	public void localTrill4Down(double x) {
		//logger.info("localTrill4Up@ " + x);
		//8 px wide
		int barWidth = 2;
		//create all bars, but at 0.0 alpha to popin later
		int trill1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth*2,0,barWidth,16), 0.0); //2px wide rect at left of area
		int trill2 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth,0,barWidth,16), 0.0);
		int trill3 = anim.startClip("simpleClip2", new Rectangle((int)x,0,barWidth,16), 0.0); 
		int trill4 = anim.startClip("simpleClip2", new Rectangle((int)x+barWidth,0,barWidth,16), 0.0); 

		//now pop in trills2-4 on time - 60,230,435,590 abs time, shift 60 for local
		anim.queueClipChange(trill4, null, null, 1.0, 0, 0, false); //popin 
		anim.queueClipChange(trill3, null, null, 1.0, 0, 170, false); //popin 
		anim.queueClipChange(trill2, null, null, 1.0, 0, 375, false); //popin 
		anim.queueClipChange(trill1, null, null, 1.0, 0, 530, false); //popin 

		//fade em all out
		anim.queueClipChange(trill1, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill2, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill3, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill4, null, null, 0.0, 500, 3500, true); //fadeout
	}

	public void localTrill4Stagger(double x) {
		//logger.info("localTrill4Up@ " + x);
		//8 px wide
		int barWidth = 2;
		//create all bars, but at 0.0 alpha to popin later
		int trill1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth*2,0,barWidth,16), 0.0); //2px wide rect at left of area
		int trill2 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth,0,barWidth,16), 0.0);
		int trill3 = anim.startClip("simpleClip2", new Rectangle((int)x,0,barWidth,16), 0.0); 
		int trill4 = anim.startClip("simpleClip2", new Rectangle((int)x+barWidth,0,barWidth,16), 0.0); 

		//now pop in trills2-4 on time - 60,230,435,590 abs time, shift 60 for local
		anim.queueClipChange(trill1, null, null, 1.0, 0, 0, false); //popin 
		anim.queueClipChange(trill3, null, null, 1.0, 0, 170, false); //popin 
		anim.queueClipChange(trill2, null, null, 1.0, 0, 375, false); //popin 
		anim.queueClipChange(trill4, null, null, 1.0, 0, 530, false); //popin 

		//fade em all out
		anim.queueClipChange(trill1, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill2, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill3, null, null, 0.0, 500, 3500, true); //fadeout
		anim.queueClipChange(trill4, null, null, 0.0, 500, 3500, true); //fadeout
	}

	public void localStabSmall(double x) {
		//logger.info("localStabA@ " + x);
		x = findNearestLight(x,true);
		//8 px wide
		int barWidth = 3;
		//create all bars, but at 0.0 alpha to popin later
		int stab1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth/2,0,barWidth,16), 1.0); //2px wide rect at left of area

		//fade out
		anim.queueClipChange(stab1, null, null, 0.0, 500, 800, true); //fadeout
	}

	public void localStabBig(double x) {
		//logger.info("localStabA@ " + x);
		x = findNearestLight(x,true);
		//8 px wide
		int barWidth = 6;
		//create all bars, but at 0.0 alpha to popin later
		int stab1 = anim.startClip("simpleClip2", new Rectangle((int)x-barWidth/2,0,barWidth,16), 1.0); //2px wide rect at left of area

		//fade out
		anim.queueClipChange(stab1, null, null, 0.0, 500, 800, true); //fadeout
	}

	/*
	 * Globals
	 */

	public void megaSparkleFaint(double x) {
		logger.info("global sparkle faint@ " + x);

		int faintSparkle = anim.startClip("sparkleClip320", new Rectangle(0,0,635,16), 0.0); // huge sparkly thing over full area
		
		//fadein NOT WORKING
		anim.queueClipChange(faintSparkle, null, null, 0.2, 4000, 500, false); //fadeout
		//fade out
		anim.queueClipChange(faintSparkle, null, null, 0.0, 2000, 12000, true); //fadeout
	}



	/**
	 * Test stuff
	 */

	public void testClip(double x)
	{
		System.out.println("PLAY AT " + x);
	}

	public void sparkleClip32(double x){
		logger.debug("sparkleClip32 started at x=" + x);
		int clip = anim.startClip("sparkleClip32", new Rectangle((int)x-16,0,32,16), 0.0);
		//anim.queueClipChange(clip, null, null, 1.0, 1700, 0, false); //make it smaller
		anim.queueClipChange(clip, null, null, 1.0, 500, 0, false); //fadein
		anim.queueClipChange(clip, null, null, 0.0, 500, 500, true); //fadeout
		//logger.info("Created Egg Sparkle at x = " + x);
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