package net.electroland.lighting.detector.animation;

import java.util.Properties;

public interface Animation
{
	/**
	 * Called by the AnimationManager to pass in initialization
	 * properties when the animation is called.  This will generally contain
	 * the contains of animation.properties.
	 * 
	 * @param props
	 */
	abstract public void init(Properties props);
	
	/**
	 * Return 'true' to notify the AnimationManager that your animation has
	 * completed it's lifecycle.
	 * @return
	 */
	abstract public boolean isDone();
	
	/**
	 * Returns the Raster to be returned for the latest frame of Animation
	 * to the AnimationManager.
	 * @return
	 */
	abstract public Raster getFrame();
}