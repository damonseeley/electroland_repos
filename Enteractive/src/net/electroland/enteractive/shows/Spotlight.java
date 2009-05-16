package net.electroland.enteractive.shows;

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
import net.electroland.enteractive.sprites.Sphere;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

/**
 * Follows the user with a feathered sphere of light which slowly fades behind them.
 * @author asiegel
 */

public class Spotlight implements Animation, SpriteListener{
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;
	private PImage sphereTexture;
	private long startTime;
	private int duration = 30000;	// milliseconds
	
	public Spotlight(Model m, Raster r, SoundManager sm, PImage sphereTexture){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.sphereTexture = sphereTexture;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
		startTime = System.currentTimeMillis();
		
		ConcurrentHashMap<Integer,Person> people = m.getPeople();
		Iterator<Person> iter = people.values().iterator();
		while(iter.hasNext()){
			Person p = iter.next();
			int[] loc = p.getLoc();
			Sphere sphere = new Sphere(spriteIndex, r, loc[0]*tileSize, loc[1]*tileSize, sm, p, sphereTexture);
			sphere.addListener(this);
			sprites.put(spriteIndex, sphere);
			spriteIndex++;
		}
	}

	public Raster getFrame() {
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			ConcurrentHashMap<Integer,Person> people = m.getPeople();
			Iterator<Person> iter = people.values().iterator();
			while(iter.hasNext()){
				Person p = iter.next();
				if(p.isNew()){
					// TODO instantiate new sprites here
					int[] loc = p.getLoc();
					Sphere sphere = new Sphere(spriteIndex, r, loc[0]*tileSize, loc[1]*tileSize, sm, p, sphereTexture);
					sphere.addListener(this);
					sprites.put(spriteIndex, sphere);
					spriteIndex++;
				}
			}
			
			Iterator<Sprite> spriteiter = sprites.values().iterator();
			while(spriteiter.hasNext()){
				Sprite sprite = (Sprite)spriteiter.next();
				sprite.draw();
			}
			raster.endDraw();
		}
		return r;
	}

	public boolean isDone() {
		if ((System.currentTimeMillis() - startTime) >= duration)
		{
			System.out.println("SPOTLIGHT IS DONE" + (System.currentTimeMillis() - startTime));
		}
		return (System.currentTimeMillis() - startTime) >= duration;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}

}