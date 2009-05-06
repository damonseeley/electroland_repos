package net.electroland.laface.shows;

import java.util.Iterator;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.laface.core.Sprite;
import net.electroland.laface.sprites.Bars;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Reflection implements Animation, TrackListener {
	
	private Raster r;
	private Vector<Track> tracks, oldtracks;
	private ConcurrentHashMap<Integer,Sprite> sprites;		// used for drawing all sprites
	private int spriteIndex = 0;
	private Bars bars;
	
	public Reflection(Raster r){
		this.r = r;
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		tracks = new Vector<Track>();
		oldtracks = new Vector<Track>();
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		bars = new Bars(spriteIndex, r, 0, 0);
		sprites.put(spriteIndex, bars);
		spriteIndex++;
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			for(Track t: tracks){
				// iterate over blobs and draw highlighted areas
				float amplitude = 0.5f;
				int wavelength = 6;
				for(Track oldt: oldtracks){
					if(t.equals(oldt)){	
						//amplitude = Math.abs(oldt.x - t.x)/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString());
						//System.out.println(oldt.x - t.x);
					}
				}
				bars.swell(t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()), amplitude, wavelength);
				//c.rect((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width, 0, 30, c.height);
				//c.image(texture, c.width-(((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width) - 50), 0, 100, c.height);
			}
			
			Iterator<Sprite> iter = sprites.values().iterator();
			while(iter.hasNext()){
				Sprite sprite = (Sprite)iter.next();
				sprite.draw(r);
			}
			c.endDraw();
		}
		return r;
	}

	public boolean isDone() {
		return false;
	}

	public void updateTracks(TrackResults results) {
		this.oldtracks = (Vector<Track>) tracks.clone();
		this.tracks = results.existing;	
	}
	
	public Bars getBars(){
		return bars;
	}

}
