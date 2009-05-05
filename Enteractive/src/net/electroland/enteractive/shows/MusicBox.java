package net.electroland.enteractive.shows;

import java.io.File;
import java.io.FileInputStream;
import java.util.HashMap;
import java.util.Iterator;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.scSoundControl.SoundNode;

/**
 * MusicBox stores locations to play back sound files when activated 
 * allowing the visitor to play with the carpet like a musical instrument.
 * @author Aaron Siegel
 */

public class MusicBox implements Animation{
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private Properties samples;
	private ConcurrentHashMap<Integer,SoundPlayer> soundPlayers;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;	// used as ID # for sprite
	
	public MusicBox(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		samples = new Properties();
		tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		soundPlayers = new ConcurrentHashMap<Integer,SoundPlayer>();
		
		try {
			samples.load(new FileInputStream(new File("./depends/musicbox.properties")));
		} catch (Exception e) {
			e.printStackTrace();
		}
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.beginDraw();
		raster.rectMode(PConstants.CENTER);
		raster.background(0);		// clear the raster
		
		synchronized (m){
			HashMap<Integer,Person> people = m.getPeople();
			synchronized(people){
				Iterator<Person> peopleiter = people.values().iterator();
				while(peopleiter.hasNext()){										// for each person...
					Person p = peopleiter.next();
					if(p.isNew()){													// if it's a new person...
						if(samples.containsKey("tile"+p.getLinearLoc())){			// if a sound event is assigned to this tile
							if(soundPlayers.containsKey(p.getLinearLoc())){
								// turn off currently playing loop
								soundPlayers.get(p.getLinearLoc()).sound.die();		// kill sound
								soundPlayers.remove(p.getLinearLoc());
							} else {
								// create a new soundNode related to this location
								SoundPlayer sp = new SoundPlayer(p);
								soundPlayers.put(p.getLinearLoc(), sp);
							}
						}
					}
				}
			}
		}
		
		raster.fill(255,0,0,255);
		raster.noStroke();
		Iterator<SoundPlayer> iter = soundPlayers.values().iterator();
		while(iter.hasNext()){
			SoundPlayer sp = iter.next();
			raster.rect(sp.person.getX()*tileSize, sp.person.getY()*tileSize, tileSize, tileSize);
			sp.update();
		}
		raster.endDraw();
		return r;
	}

	public void cleanUp() {
	}

	public boolean isDone() {
		return false;
	}
	
	
	
	
	
	
	/**
	 * Stores reference to Person object that created it as well
	 * as the sound node currently being played back. When either
	 * of them die, call back to MusicBox to remove it from the CHM.
	 */
	
	public class SoundPlayer{
		
		int id;
		Person person;
		SoundNode sound;
		boolean looping;
		
		public SoundPlayer(Person person){
			this.id = person.getLinearLoc();
			this.person = person;
			String[] props = samples.getProperty("tile"+id).split(",");
			sound = sm.createMonoSound(props[0], 0.5f, 0, 1, 1);
			if(Boolean.parseBoolean(props[1])){
				sound.set_looping(true);
			}
		}
		
		public void update(){
			if(person.isDead()){
				if(!looping){
					soundPlayers.remove(person.getLinearLoc());
				}
			} else if(sound != null && !sound.isAlive()){
				soundPlayers.remove(person.getLinearLoc());
			}
		}
	}

}
