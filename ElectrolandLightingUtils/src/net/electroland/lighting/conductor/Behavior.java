package net.electroland.lighting.conductor;

import net.electroland.input.InputDeviceListener;
import net.electroland.lighting.detector.DetectorManager;
import net.electroland.lighting.detector.animation.AnimationListener;
import net.electroland.lighting.detector.animation.AnimationManager;

abstract public class Behavior implements InputDeviceListener, AnimationListener{

	private AnimationManager am;
	private DetectorManager dm;
	
	abstract public int getPriority();
	
	final public void setAnimationManager(AnimationManager am)
	{
		this.am = am;
	}
	public AnimationManager getAnimationManager()
	{
		return am;
	}
	final public void setDetectorManager(DetectorManager dm)
	{
		this.dm = dm;
	}
	public DetectorManager getDetectorManger()
	{
		return dm;
	}


	// would be MUCH improved if Behaviors didn't need direct access to
	// AnimationManager or DetectorManager.  e.g., you call "startAnimation(...)"
}