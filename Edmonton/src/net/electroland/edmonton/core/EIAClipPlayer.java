package net.electroland.edmonton.core;

import java.awt.Rectangle;

import org.apache.log4j.Logger;

import net.electroland.ea.AnimationManager;
import net.electroland.edmonton.core.sequencing.ClipCue;

public class EIAClipPlayer {

	private AnimationManager anim;
	
    static Logger logger = Logger.getLogger(EIAClipPlayer.class);
	
	public EIAClipPlayer(AnimationManager am)
	{
		this.anim = am;
	}

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
	
	/*
	 * EIA Show methods
	 */
	
	public void introLocalTrill4Up(double x) {
		logger.info("localTrill4Up@ " + x);
		//8 px wide, ms times 15 176 322 510, where 15 is effectively 0 in cue time

		//create the first bar and have it appear immediately
		int trill1 = anim.startClip("simpleClip2", new Rectangle((int)x-4,0,2,16), 1.0); //2px wide rect at left of area
		//create the other bars, but at 0.0 alpha to popin later
		int trill2 = anim.startClip("simpleClip2", new Rectangle((int)x-2,0,2,16), 0.0);
		int trill3 = anim.startClip("simpleClip2", new Rectangle((int)x,0,2,16), 0.0); 
		int trill4 = anim.startClip("simpleClip2", new Rectangle((int)x+2,0,2,16), 0.0); 
		
		//now pop in trills2-4 on time
		anim.queueClipChange(trill2, null, null, 1.0, 0, 161, false); //popin 161
		anim.queueClipChange(trill3, null, null, 1.0, 0, 307, false); //popin 307
		anim.queueClipChange(trill4, null, null, 1.0, 0, 495, false); //popin 495
 
		
		//fade em all out
		anim.queueClipChange(trill1, null, null, 0.0, 500, 1500, true); //fadeout
		anim.queueClipChange(trill2, null, null, 0.0, 500, 1500, true); //fadeout
		anim.queueClipChange(trill3, null, null, 0.0, 500, 1500, true); //fadeout
		anim.queueClipChange(trill4, null, null, 0.0, 500, 1500, true); //fadeout
		
		
	}
	
	
	
	
	/**
	 * Local Util Methods
	 */
	
	private void LocalNoteTrill(double x, double width, double numNotes, int[] times, boolean LtoR){
		
		
		
	}
	
}