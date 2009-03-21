package net.electroland.enteractive.shows;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.enteractive.sprites.Pad;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

/**
 * LilyPad creates sprites at specific tiles, that when activated by users, trigger playful animations.
 * @author asiegel
 */

public class LilyPad implements Animation, SpriteListener {
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private List<Sprite> sprites;
	private List<Pad> pads;
	private int padCount = 0;		// current number
	private int maxPads = 7;		// maximum pads at any time
	private int padDelay = 5;		// mandatory delay between adding new pads
	private int delayCount = 0;
	private float padOdds = 0.8f;	// odds of creating a new pad
	
	public LilyPad(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ArrayList<Sprite>();
		pads = new ArrayList<Pad>();
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		if(sprites.size() < maxPads){			// if not maxed out on pads...
			if(delayCount < padDelay){			// must reach mandatory delay (in frames)
				delayCount++;
			} else {
				if(Math.random() > padOdds){	// chance of creating a new pad
					Pad pad = new Pad(r, (int)(Math.random()*15)+1, (int)(Math.random()*10)+1, 0, 150, 1000);
					sprites.add(pad);
					pads.add(pad);
					delayCount = 0;
					padCount++;
				}
			}
		}
		
		synchronized (m){
			// TODO check Enter events in the model to see if any pads are triggered
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			Iterator iter = sprites.iterator();
			while(iter.hasNext()){
				Sprite sprite = (Sprite)iter.next();
				sprite.draw();
			}
			raster.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.beginDraw();
		raster.background(0);			// clear the raster
		raster.endDraw();
	}

	public boolean isDone() {
		return false;
	}

	public void spriteComplete(Sprite sprite) {
		// TODO Create a new LilyPad
	}

}
