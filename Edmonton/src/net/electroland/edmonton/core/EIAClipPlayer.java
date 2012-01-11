package net.electroland.edmonton.core;

import java.awt.Rectangle;

import net.electroland.ea.AnimationManager;

public class EIAClipPlayer {

	private AnimationManager anim;
	
	public EIAClipPlayer(AnimationManager am)
	{
		this.anim = am;
	}

	public void testClip(double x)
	{
		System.out.println("PLAY AT " + x);
	}
	
	public void sparkleClip32(double x){
		int clip = anim.startClip("sparkleClip32", new Rectangle((int)x-16,0,32,16), 0.0);
		//anim.queueClipChange(clip, null, null, 1.0, 1700, 0, false); //make it smaller
		anim.queueClipChange(clip, null, null, 1.0, 500, 0, false); //fadein
		anim.queueClipChange(clip, null, null, 0.0, 500, 500, true); //fadeout
		//logger.info("Created Egg Sparkle at x = " + x);
	}
	
	
}