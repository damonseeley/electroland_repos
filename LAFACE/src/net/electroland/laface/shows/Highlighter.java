package net.electroland.laface.shows;

import java.util.Vector;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;
import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Highlighter implements Animation, TrackListener {

	private Raster r;
	private Vector<Track> tracks;
	private PImage texture;
	
	public Highlighter(Raster r, PImage texture){
		this.r = r;
		this.tracks = new Vector<Track>();
		this.texture = texture;
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);	// fully bright during the day
			//System.out.println(tracks.size());
			/*
			// TODO only do this when it's dark enough to see (dusk till dawn)
			c.background(0);
			c.noStroke();
			c.fill(255);
			for(Track t: tracks){
				// iterate over blobs and draw highlighted areas
				//c.rect((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width, 0, 30, c.height);
				c.image(texture, c.width-(((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width) - 50), 0, 100, c.height);
			}
			*/
			c.endDraw();
		}
		return r;
	}
	
	public void cleanUp() {
		PGraphics myRaster = (PGraphics)(r.getRaster());
		myRaster.beginDraw();
		myRaster.background(0);
		myRaster.endDraw();
	}

	public boolean isDone() {
		return false;
	}

	public void updateTracks(TrackResults results) {
		this.tracks = results.existing;		
	}

}
