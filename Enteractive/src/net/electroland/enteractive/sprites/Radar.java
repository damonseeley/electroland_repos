package net.electroland.enteractive.sprites;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.shows.LilyPad;
import net.electroland.lighting.detector.animation.Raster;

public class Radar extends Sprite{
	
	private LilyPad show;
	private PImage texture;
	private int radius, rotSpeed, rotation, timeOut, alpha, fadeSpeed;
	private long startTime, radarStartTime, fadeStart;
	private boolean fadeOut = false;
	private ConcurrentHashMap<Integer,RadarTarget> targets;
	private Person person;

	public Radar(int id, Raster raster, float x, float y, SoundManager sm, LilyPad show, Person person, PImage texture, int radius, int rotSpeed) {
		super(id, raster, x, y, sm);
		this.show = show;
		this.person = person;
		this.texture = texture;
		this.radius = radius*tileSize;
		this.rotSpeed = rotSpeed;
		alpha = 255;
		timeOut = rotSpeed*3;	// 3 sweeps before dying
		fadeSpeed = rotSpeed/2;
		targets = new ConcurrentHashMap<Integer,RadarTarget>();
		radarStartTime = startTime = System.currentTimeMillis();
	}

	@Override
	public void draw() {
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.translate(x, y);
			c.rotate((float)(Math.PI/180)*rotation);
			c.tint(255,255,255,alpha);
			c.image(texture, 0-radius, 0-radius, radius*2, radius*2);
			c.popMatrix();
		}

		// check for collision against a person
		ConcurrentHashMap<Integer,Person> people = show.m.getPeople();
		synchronized(people){
			Iterator<Person> iter = people.values().iterator();
			while(iter.hasNext()){
				Person newperson = iter.next();
				// if not already a target and not the person activating the radar...
				if(!targets.containsKey(newperson.getID()) && newperson.getID() != person.getID()){
					float xdiff = x - newperson.getX()*tileSize;
					float ydiff = y - newperson.getY()*tileSize;
					float hypo = (float)Math.sqrt(xdiff*xdiff + ydiff*ydiff);
					if(hypo < radius){		// check if distance to indicator is less than radius
						targets.put(newperson.getID(), new RadarTarget(newperson));
					}
				}
			}
		}
		
		Iterator<RadarTarget> targetiter = targets.values().iterator();
		while(targetiter.hasNext()){
			RadarTarget rt = targetiter.next();
			// compare theta of person to rotation of radar
			float xdiff = x - rt.person.getX()*tileSize;
			float ydiff = y - rt.person.getY()*tileSize;
			float angle = (float)(Math.atan(xdiff/ydiff)/(Math.PI/180));
			if(ydiff < 0){
				angle += 180;
			}
			if(xdiff >=0 && ydiff<0){
				angle += 360;
			}
			// play sound if they match up
			if(rotation >= angle && !rt.played){
				if(raster.isProcessing()){
					PGraphics c = (PGraphics)canvas;
					sm.createMonoSound(sm.soundProps.getProperty("radar"), c.width/2, y, c.width, c.height);
				}
				rt.played = true;
			}
			 
		}
		
		if(System.currentTimeMillis() - startTime < rotSpeed){
			// keep rotating
			rotation = (int)(((System.currentTimeMillis() - startTime) / (float)rotSpeed) * 360);
		} else {
			// reset rotation and reset radar targets to allow them to play sounds again
			startTime = System.currentTimeMillis();
			Iterator<RadarTarget> newiter = targets.values().iterator();
			while(newiter.hasNext()){
				RadarTarget rt = newiter.next();
				rt.played = false;
			}
		}
		
		if(System.currentTimeMillis() - radarStartTime > timeOut || person.isDead()){
			fadeOut = true;
			fadeStart = System.currentTimeMillis();
		}
		
		if(fadeOut){
			alpha = 255 - (int)(((System.currentTimeMillis() - fadeStart) / (float)fadeSpeed) * 255);
			if(alpha <= 0){
				die();
			}
		}
	}
	
	
	
	
	/**
	 * Holds an indicator object which contains a person and sound info.
	 */
	
	public class RadarTarget{
		
		Person person;
		boolean played;
		
		public RadarTarget(Person person){
			this.person = person;
			played = false;
		}
		
		public void update(){
			if(person.isDead()){
				targets.remove(person.getID());
			}
		}
	}

}
