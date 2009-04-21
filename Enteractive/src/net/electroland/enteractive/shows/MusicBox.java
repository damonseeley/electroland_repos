package net.electroland.enteractive.shows;

import java.util.HashMap;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class MusicBox implements Animation{
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;	// used as ID # for sprite
	
	public MusicBox(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
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
						// TODO find if a sound event is assigned to this tile
						// TODO if so, create a new soundNode related to this location
					}
				}
			}
		}
		return r;
	}

	public void cleanUp() {
	}

	public boolean isDone() {
		return false;
	}

}
