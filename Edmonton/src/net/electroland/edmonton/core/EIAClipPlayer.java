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
	
	
}