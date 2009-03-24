package net.electroland.enteractive.shows;

import java.util.HashMap;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.enteractive.sprites.ExplodingCross;
import net.electroland.enteractive.sprites.ImageSprite;
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
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private ConcurrentHashMap<Integer,Pad> pads;
	private int spriteIndex = 0;	// used as ID # for sprite
	private int maxPads = 7;		// maximum pads at any time
	private int padDelay = 5;		// mandatory delay between adding new pads
	private int delayCount = 0;
	private float padOdds = 0.8f;	// odds of creating a new pad
	private PImage rippleTexture;	// PNG image for ripple sprite
	
	public LilyPad(Model m, Raster r, SoundManager sm, PImage rippleTexture){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.rippleTexture = rippleTexture;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		pads = new ConcurrentHashMap<Integer,Pad>();
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		if(pads.size() < maxPads){			// if not maxed out on pads...
			if(delayCount < padDelay){			// must reach mandatory delay (in frames)
				delayCount++;
			} else {
				if(Math.random() > padOdds){	// chance of creating a new pad
					Pad pad = new Pad(spriteIndex, r, (int)(Math.random()*15)+1, (int)(Math.random()*10)+1, 0, 150, 1000);
					pad.addListener(this);
					sprites.put(spriteIndex, pad);
					pads.put(spriteIndex, pad);
					delayCount = 0;
					spriteIndex++;
				}
			}
		}
		
		synchronized (m){
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster

			HashMap<Integer,Person> people = m.getPeople();
			synchronized(people){
				Iterator<Person> peopleiter = people.values().iterator();
				while(peopleiter.hasNext()){										// for each person...
					Person p = peopleiter.next();
					if(p.isNew()){													// if it's a new person...
						Iterator<Pad> i = pads.values().iterator();
						while(i.hasNext()){											// check every active pad
							Pad pad = i.next();
							if(pad.getX() == p.getX() && pad.getY() == p.getY()){	// if new person on the pad...
								pads.remove(pad.getID());
								// create new action sprite here
								int luckyNumber = (int)(Math.random()*2 - 0.01);
								//System.out.println(luckyNumber);
								Sprite sprite = null;
								switch(luckyNumber){
									case 0:
										sprite = new ImageSprite(spriteIndex, r, pad.getX(), pad.getY(), rippleTexture, 0.1f, 0.1f);
										break;
									case 1:
										sprite = new ExplodingCross(spriteIndex, r, (int)pad.getX(), (int)pad.getY(), 1500);
										break;
								}
								if(sprite != null){
									sprite.addListener(this);
									sprites.put(spriteIndex, sprite);
									spriteIndex++;
								}
								pad.die();
							}
						}
					}
				}
			}
			
			Iterator<Sprite> iter = sprites.values().iterator();
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
		sprites.remove(sprite.getID());
		//System.out.println("sprite "+sprite.getID()+" removed");
	}

}
