package net.electroland.enteractive.shows;

import java.util.ConcurrentModificationException;
import java.util.HashMap;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.enteractive.sprites.Cross;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class ExampleAnimation implements Animation, SpriteListener {

	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;
	private int cycles = 90;
	
	public ExampleAnimation(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
	}

	public void initialize() {
		// TODO play some sound
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	
	public Raster getFrame() {
		// lock the model so other people don't modify it while we do paint
		// calls based on it.
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics myRaster = (PGraphics)(r.getRaster());
			myRaster.beginDraw();
		
			
			// TODO THIS SHOULD BE THE NORMAL WAY TO FIND NEW PEOPLE/CREATE NEW SPRITES
			HashMap<Integer,Person> people = m.getPeople();
			Iterator<Person> iter = people.values().iterator();
			while(iter.hasNext()){
				Person p = iter.next();
				if(p.isNew()){
					// TODO instantiate new sprites here
					Cross cross = new Cross(spriteIndex, r, p, 1, 1, sm, 3, 3);		// 3x3 cross
					int[] loc = p.getLoc();
					//System.out.println(loc[0]+" "+loc[1]);
					cross.moveTo(loc[0]*tileSize, loc[1]*tileSize);
					cross.addListener(this);
					//sprites.add(cross);
					sprites.put(spriteIndex, cross);
					spriteIndex++;
				}
			}
			
			
			/*
			myRaster.background(0);		// clear the raster
			Cross cross = new Cross(r, null, 1, 1, 3, 3);		// 3x3 cross
			boolean[] sensorlist = m.getSensors();
			for(int i=0; i<sensorlist.length; i++){	// sensorlist is 16x11
				if(sensorlist[i]){
					int x = i % 16;			// probably shouldn't be static values
					int y = i / 16;
					// position is offset by 1 because of the extra column on each side
					cross.moveTo((x+1)*tileSize, y*tileSize);	// moves instance of sprite to active tile
					cross.draw();			// draws instance
				} 
			}
			*/
			
			
			
			
			// TODO THIS SHOULD BE THE NORMAL WAY TO DRAW ALL SPRITES
			try{
				myRaster.background(0);		// clear the raster
				Iterator<Sprite> spriteiter = sprites.values().iterator();
				while(spriteiter.hasNext()){
					Sprite sprite = (Sprite)spriteiter.next();
					sprite.draw();
				}
			}catch(ConcurrentModificationException e){
				// TODO WHY DOES IT THROW THESE ERRORS?
				e.printStackTrace();
			}
			
			
			
			
			//myRaster.background(255,0,0); // FULLY ON
			myRaster.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		// play some sound, clear the raster, etc.
		//System.out.println("cleaning up.");
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return cycles-- <= 0; // no timeout for now
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}
}