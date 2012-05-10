package net.electroland.edmonton.core;

import net.electroland.ea.AnimationManager;
import net.electroland.utils.lighting.ELUManager;

import org.apache.log4j.Logger;

public class EIAClipPlayer2 extends EIAClipPlayer {

	static Logger logger = Logger.getLogger(EIAClipPlayer2.class);

	public EIAClipPlayer2(AnimationManager am, ELUManager elu, SoundController sc)
	{
		super(am, elu, sc);
	}

	public void test(double x){
		logger.info("we're on at " + x);
	}
}