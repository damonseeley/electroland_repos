package net.electroland.edmonton.core;

import net.electroland.ea.AnimationManager;

public class EIAClipPlayer {

	private AnimationManager am;
	
	public EIAClipPlayer(AnimationManager am)
	{
		this.am = am;
	}

	public void testClip(double x)
	{
		System.out.println("PLAY AT " + x);
	}
}