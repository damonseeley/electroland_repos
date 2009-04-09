package net.electroland.noho.core;

import java.awt.image.BufferedImage;


public abstract class TextAnimation {
	
	private boolean isDestroyed = false;
	
	public boolean complete;
	

	public TextAnimation() {
	}

	
	protected abstract void process(); 
	
	public abstract BufferedImage getImage();

	
	public void destroy() {
		isDestroyed = true;
	}
	
	public boolean isDestroyed() {
		return isDestroyed;
	}

}
