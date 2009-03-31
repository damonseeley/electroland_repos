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
import net.electroland.enteractive.sprites.BullsEye;
import net.electroland.enteractive.sprites.ExplodingCross;
import net.electroland.enteractive.sprites.Pad;
import net.electroland.enteractive.sprites.Propeller;
import net.electroland.enteractive.sprites.Ripple;
import net.electroland.enteractive.sprites.Shooter;
import net.electroland.enteractive.sprites.Single;
import net.electroland.enteractive.sprites.Sparkler;
import net.electroland.enteractive.sprites.Spiral;
import net.electroland.enteractive.sprites.Sweep;
import net.electroland.enteractive.sprites.TickerBox;
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
	private ConcurrentHashMap<Integer,Single> billiejean;
	private int spriteIndex = 0;	// used as ID # for sprite
	private int maxPads = 7;		// maximum pads at any time
	private int padDelay = 5;		// mandatory delay between adding new pads
	private int delayCount = 0;
	private float padOdds = 0.8f;	// odds of creating a new pad
	private PImage rippleTexture;	// PNG image for ripple sprite
	private PImage sweepTexture;
	private PImage propellerTexture;
	private PImage spiralTexture;
	private PImage sphereTexture;
	
	public LilyPad(Model m, Raster r, SoundManager sm, PImage rippleTexture, PImage sweepTexture, PImage propellerTexture, PImage spiralTexture, PImage sphereTexture){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.rippleTexture = rippleTexture;
		this.sweepTexture = sweepTexture;
		this.propellerTexture = propellerTexture;
		this.spiralTexture = spiralTexture;
		this.sphereTexture = sphereTexture;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		pads = new ConcurrentHashMap<Integer,Pad>();
		billiejean = new ConcurrentHashMap<Integer,Single>();
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
					//Pad pad = new Pad(spriteIndex, r, (int)Math.floor(Math.random()*15.99f)+1, (int)Math.floor(Math.random()*9.99f)+1, sm, 0, 255, 500);
					Pad pad = new Pad(spriteIndex, r, (int)Math.floor(Math.random()*13.99f)+2, (int)Math.floor(Math.random()*7.99f)+2, sm, 0, 255, 500);
					//pad.addListener(this);
					//sprites.put(spriteIndex, pad);
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
						int[] loc = p.getLoc();
						Single single = new Single(spriteIndex, r, p, loc[0]*tileSize, loc[1]*tileSize, sm);	// single tile sprite (billie jean mode)
						single.addListener(this);
						billiejean.put(spriteIndex, single);
						spriteIndex++;
						
						if(loc[0] == 1){			// near the entrance
							Shooter shooter = new Shooter(spriteIndex, r, 0, (int)loc[1]*tileSize, sm, sweepTexture, 1000, false);
							shooter.addListener(this);
							sprites.put(spriteIndex, shooter);
							spriteIndex++;
						} else if(loc[0] == 16){	// near the sidewalk
							Shooter shooter = new Shooter(spriteIndex, r, 18*tileSize, (int)loc[1]*tileSize, sm, sweepTexture, 1000, true);
							shooter.addListener(this);
							sprites.put(spriteIndex, shooter);
							spriteIndex++;
						}
						
						Iterator<Pad> i = pads.values().iterator();
						while(i.hasNext()){											// check every active pad
							Pad pad = i.next();
							if(pad.getX() == p.getX() && pad.getY() == p.getY()){	// if new person on the pad...
								//System.out.println("pad "+ pad.getID() +" activated");
								pads.remove(pad.getID());
								// create new action sprite here
								//System.out.println(luckyNumber);
								
								Sprite sprite = null;
								/*
								// TODO this conditional needs to be removed
								if(pad.getX() == 1){	// if near entrance
									int luckyNumber = (int)(Math.random()*3 - 0.01);
									if(luckyNumber < 0){
										luckyNumber = 0;
									}
									switch(luckyNumber){
										case 0:
											sprite = new Ripple(spriteIndex, r, pad.getX(), pad.getY(), sm, rippleTexture);
											break;
										case 1:
											sprite = new Sweep(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, sweepTexture, 1500, false);
											break;
										case 2:
											sprite = new Shooter(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, sweepTexture, 1000, false);
											break;
									}
								} else if(pad.getX() == 16){	// if near the sidewalk
									int luckyNumber = (int)(Math.random()*3 - 0.01);
									if(luckyNumber < 0){
										luckyNumber = 0;
									}
									switch(luckyNumber){
										case 0:
											sprite = new Ripple(spriteIndex, r, pad.getX(), pad.getY(), sm, rippleTexture);
											break;
										case 1:
											sprite = new Sweep(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, sweepTexture, 1500, true);
											break;
										case 2:
											sprite = new Shooter(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, sweepTexture, 1000, true);
											break;
									}
								} else {	// anywhere in the middle
								*/
									int luckyNumber = (int)(Math.random()*6 - 0.01); // temporarily omitting sparkler
									if(luckyNumber < 0){
										luckyNumber = 0;
									}
									switch(luckyNumber){
										case 0:
											sprite = new Ripple(spriteIndex, r, pad.getX(), pad.getY(), sm, rippleTexture);
											break;
										case 1:
											sprite = new ExplodingCross(spriteIndex, r, (int)pad.getX(), (int)pad.getY(), sm, 2250);
											break;
										case 2:
											sprite = new TickerBox(spriteIndex, r, p, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, 2000);
											break;
										case 3:
											sprite = new Propeller(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, p, propellerTexture);
											break;
										case 4:
											sprite = new BullsEye(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, p, 3, 2000);
											break;
										case 5:
											sprite = new Spiral(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, p, spiralTexture);
											break;
										case 6:
											sprite = new Sparkler(spriteIndex, r, (int)pad.getX()*tileSize, (int)pad.getY()*tileSize, sm, p, sphereTexture);
											break;
									}
								//}
								//if(sprite != null){
								sprite.addListener(this);
								sprites.put(spriteIndex, sprite);
								spriteIndex++;
								//}
								
							}
						}
					}
				}
			}
			
			Iterator<Single> singleiter = billiejean.values().iterator();
			while(singleiter.hasNext()){
				Sprite sprite = (Sprite)singleiter.next();
				sprite.draw();
			}
			
			Iterator<Pad> paditer = pads.values().iterator();
			while(paditer.hasNext()){
				Sprite sprite = (Sprite)paditer.next();
				sprite.draw();
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
		if(sprite instanceof Single){
			billiejean.remove(sprite.getID());
		} else {
			sprites.remove(sprite.getID());
		}
		
		//if(!sprites.containsKey(sprite.getID())){
			//System.out.println("sprite "+sprite.getID()+" removed");
		//}
	}

}
