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
	private ConcurrentHashMap<Integer,Ghost> ghosts;
	private int ghostCount = 10;
	//private ConcurrentHashMap<Integer,Sprite> sprites;
	//private int spriteIndex = 0;
	private PImage sphereTexture;
	//private int duration = 30000;	// milliseconds
	
	public Spotlight(Model m, Raster r, SoundManager sm, PImage sphereTexture){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.sphereTexture = sphereTexture;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		//sprites = new ConcurrentHashMap<Integer,Sprite>();
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
		ghosts = new ConcurrentHashMap<Integer,Ghost>();
		for(int i=0; i<ghostCount; i++){
			Ghost g =  new Ghost((float)(Math.random()*raster.width), (float)(Math.random()*raster.height));
			g.changeVector(raster);
			ghosts.put(i, g);
		}
		/*
		// TODO this is all for tracking people, not used when in screensaver mode
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
		*/
	}

	public Raster getFrame() {
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			Iterator<Ghost> iter = ghosts.values().iterator();
			while(iter.hasNext()){
				Ghost g = iter.next();
				g.draw(raster);
			}
			/*
			// TODO this is all for tracking people, not used when in screensaver mode
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
			*/
			raster.endDraw();
		}
		return r;
	}

	public boolean isDone() {
		/*
		if ((System.currentTimeMillis() - startTime) >= duration)
		{
			System.out.println("SPOTLIGHT IS DONE" + (System.currentTimeMillis() - startTime));
		}
		return (System.currentTimeMillis() - startTime) >= duration;
		*/
		return false;
	}

	public void spriteComplete(Sprite sprite) {
		//sprites.remove(sprite.getID());
	}
	
	
	
	
	
	
	public class Ghost{
		private float x, y, xdiff, ydiff, hypo;
		private int xtarget, ytarget, xstart, ystart;
		private float damping = 0.1f;
		private int diameter, alpha;
		private long startTime;
		private int vectorDuration;
		private int minDuration = 5000;
		private int maxDuration = 12000;
		private boolean speedUp = false;	// TODO will switch in the future
		
		public Ghost(float x, float y){
			this.x = x;
			this.y = y;
			this.xstart = (int)x;
			this.ystart = (int)y;
			diameter = tileSize * (int)((Math.random()*5)+5);
			alpha = (int)(Math.random()*32) + 32;
			damping = (float)(Math.random()*0.1f + 0.02f);
			vectorDuration = (int)(Math.random()*(maxDuration - minDuration)) + minDuration;
			startTime = System.currentTimeMillis();
		}
		
		public void draw(PGraphics raster){
			//if(System.currentTimeMillis() - startTime > vectorDuration){
				//changeVector(raster);
			//}
			if(speedUp){
				// dampen towards middle of distance increasing speed
				xdiff = xtarget - x;
				ydiff = ytarget - y;
				if(Math.abs(xdiff) < Math.abs(xtarget - xstart)/2 || Math.abs(ydiff) < Math.abs(ytarget - ystart)/2){
					speedUp = false;
					//System.out.println("SLOW DOWN");
				} else {
					if((x-xstart) == 0 && (y-ystart) == 0){
						hypo = (float)Math.sqrt((xdiff*xdiff) + (ydiff*ydiff));
						x += xdiff/hypo;
						y += ydiff/hypo;
					} else {
						//x += (x-xstart) / (1-damping);
						//y += (y-ystart) / (1-damping);
						x = xstart + ((x-xstart) / (1-damping));
						y = ystart + ((y-ystart) / (1-damping));
						//System.out.println((x-xstart) / (1-damping) +" "+ (y-ystart) / (1-damping));
					}
				}
			} else {
				// dampen towards final target decreasing speed
				if(x != xtarget || y != ytarget){
					xdiff = xtarget - x;
					ydiff = ytarget - y;
					hypo = (float)Math.sqrt((xdiff*xdiff) + (ydiff*ydiff));
					if(Math.abs(hypo) < 1){
						x = xtarget;
						y = ytarget;
						changeVector(raster);
					} else {
						x += xdiff*damping;
						y += ydiff*damping;
					}
				}
			}
			// this is just in case shit goes crazy and they go way out of frame
			if((x < -100 || x > raster.width+100) || (y < -100 || y > raster.height+100)){
				//System.out.println(x+" "+y);
				x = (int)(Math.random()*raster.width);
				y = (int)(Math.random()*raster.height);
				changeVector(raster);
			}
			raster.tint(255,255,255,alpha);
			raster.image(sphereTexture, x-(diameter/2), y-(diameter/2), diameter, diameter);
		}
		
		public void changeVector(PGraphics raster){
			xtarget = (int)(Math.random()*raster.width);
			ytarget = (int)(Math.random()*raster.height);
			xstart = (int)x;
			ystart = (int)y;
			speedUp = true;
			vectorDuration = (int)(Math.random()*(maxDuration - minDuration)) + minDuration;
			startTime = System.currentTimeMillis();
			//System.out.println(xtarget+" "+ytarget);
		}
	}

}