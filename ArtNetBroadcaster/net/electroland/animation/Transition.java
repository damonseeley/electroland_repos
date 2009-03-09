package net.electroland.animation;

abstract class Transition implements Animation {
	
	abstract Raster getFrame(Animation one, Animation two);
}