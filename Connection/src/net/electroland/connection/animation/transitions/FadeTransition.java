package net.electroland.connection.animation.transitions;

import net.electroland.connection.animation.Animation;
import net.electroland.connection.animation.Transition;

public class FadeTransition extends Transition {

	private double progress, speed;
	
	public FadeTransition(Animation startAnimation, Animation endAnimation){
		super(startAnimation, endAnimation);
		progress = 0.0;
		speed = 0.03;
	}

	public byte[] draw() {

		// blend the pixels from start and end
		byte[] startPix = this.getStartAnimationPixels();
		byte[] endPix = this.getEndAnimationPixels();

		if (startPix.length != endPix.length){
			throw new RuntimeException("startPix.length != endPix.length");
		}

		byte[] myPix = new byte[startPix.length];
		
		for (int i = 0; i < startPix.length; i++){
			myPix[i] = (byte)((progress * (int)(endPix[i] & 0xFF)) + ((1.0 - progress) * (int)(startPix[i] & 0xFF)));
		}
		
		progress += speed;
		
		if (progress >= 1.0){
			super.complete();
			progress = 1.0;
		}

		return myPix;
	}

	public int getDefaultDuration(){
		return getEndAnimation().getDefaultDuration();
	}
	
	public void start() {
		System.out.println("START: FadeTransition");
		// setting this to 0 means that we'll always return the start animation's
		// pix
		progress = 0.0;
	}

	public void stop() {
		System.out.println("STOP: FadeTransition");
		// setting this to 100.0 means that we'll always return the end animation's
		// pix
		progress = 1.0;
	}
}
