package net.electroland.animation;

interface Transition extends Animation {

	// return some transitional frame between one and two, relating to an
	// internal sense of state.
	public Raster getFrame(Animation one, Animation two);

	// when Transition.isDone() is called, AnimationManager will dump the 
	// Transition and replace it with the Animation returned by this method.
	// special case: if getEndAnimation().isDone() == true, there is nothing
	// left to do.  (this condition should be checked in AnimationManager)
	public Animation getEndAnimation();
}