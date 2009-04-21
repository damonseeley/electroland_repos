package net.electroland.enteractive.sprites;

import java.util.Iterator;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.lighting.detector.animation.Raster;

public class Shooter extends Sprite {

	private PImage image;
	private int duration, fadeDuration;
	private long startTime, fadeStartTime;
	private boolean switchDirection, fadeOut;
	private int sweepLength;
	private int alpha;
	private LilyPad show;
	
	public Shooter(int id, Raster raster, float x, float y, SoundManager sm, PImage image, int duration, boolean switchDirection, LilyPad show) {
		super(id, raster, x, y, sm);
		this.image = image;
		this.duration = duration;
		this.switchDirection = switchDirection;
		this.show = show;
		sweepLength = 150;
		alpha = 255;
		fadeDuration = 250;
		startTime = System.currentTimeMillis();
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			sm.createMonoSound(sm.soundProps.getProperty("shooter"), x, y, c.width, c.height);
		}
		fadeOut = false;
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.tint(255,255,255,alpha);
			if(switchDirection){
				if(!fadeOut){
					x = c.width - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
					if(x <= 0-sweepLength){
						die();
					}
				}
				c.translate(x, y);
				c.rotate((float)Math.PI);	// flip it
				c.image(image, 0-sweepLength,  0-(tileSize/2), sweepLength, tileSize);
			} else {
				if(!fadeOut){
					x = (int)(((System.currentTimeMillis() - startTime) / (float)duration) * (c.width+sweepLength));
					if(x >= c.width+sweepLength){
						die();
					}
				}
				c.image(image, x-sweepLength, y-(tileSize/2), sweepLength, tileSize);
			}
			if(fadeOut){
				alpha = 255 - (int)(((System.currentTimeMillis() - fadeStartTime) / fadeDuration) * 255);
				if(alpha <= 0){
					die();
				}
			}
			c.tint(255,255,255,255);	// set back to opaque, since processing has a bug with tint
			c.popMatrix();
		}
		
		if(!fadeOut){
			// check for collision against an indicator sprite
			Iterator<Single> singleiter = show.billiejean.values().iterator();
			while(singleiter.hasNext()){
				Sprite sprite = (Sprite)singleiter.next();
				if(System.currentTimeMillis() - startTime > duration/4){
					if((int)Math.floor(sprite.getX()/tileSize) == (int)Math.floor(x/tileSize)){		// if sharing the same X pos...
						if((int)Math.floor(sprite.getY()/tileSize) == (int)Math.floor(y/tileSize)){	// and sharing same Y pos...
							fadeStartTime = System.currentTimeMillis();
							fadeOut = true;
							show.addRipple((int)Math.floor(x/tileSize), (int)Math.floor(y/tileSize));	// explosion effect
						}
					}
				}
			}
		}
	}

}
