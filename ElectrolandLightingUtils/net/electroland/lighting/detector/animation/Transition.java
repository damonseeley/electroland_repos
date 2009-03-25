package net.electroland.lighting.detector.animation;

interface Transition extends Completable{
	public Raster getFrame(Animation startAnimation, Animation finishAnimation);
}