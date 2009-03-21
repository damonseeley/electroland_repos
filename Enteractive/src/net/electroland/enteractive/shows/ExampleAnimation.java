package net.electroland.enteractive.shows;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.Person;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.sprites.Cross;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class ExampleAnimation implements Animation {

	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	//private int cycles = 600;
	private List<Sprite> sprites;
	
	public ExampleAnimation(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ArrayList<Sprite>();
		//System.out.println("instantiated.");
	}

	public void initialize() {
		// play some sound, clear the raster, etc.
		//System.out.println("initializing.");
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
			myRaster.background(0);		// clear the raster
			
			/*
			// TODO PRETTY SURE THIS IS CAUSING THREAD LOCKING
			
			HashMap<Integer,Person> enters = m.getEnters();
			Iterator<Person> iter = enters.values().iterator();
			while(iter.hasNext()){
				// TODO instantiate new sprites here
			}
			//m.clearEnters();	// always clear to prevent double instances
			
			HashMap<Integer,Person> exits = m.getExits();
			Iterator<Person> exititer = exits.values().iterator();
			while(exititer.hasNext()){
				// TODO destroy sprites here
			}
			//m.clearExits();	// always clear to prevent double removal attempts
			*/
			
			Cross cross = new Cross(r, 1, 1, 3, 3);		// 3x3 cross
			cross.moveTo(9,6);
			cross.draw();
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
			
			
			/*
			// this is the normal way to draw all sprites
			Iterator iter = sprites.iterator();
			while(iter.hasNext()){
				Sprite sprite = (Sprite)iter.next();
				sprite.draw();
			}
			*/
			
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
		return false;//cycles-- <= 0; // no timeout for now
	}
}