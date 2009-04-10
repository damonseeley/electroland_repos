package net.electroland.laface.shows;

import java.util.Vector;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.blobDetection.match.Track;
import net.electroland.blobDetection.match.TrackListener;
import net.electroland.blobDetection.match.TrackResults;
import net.electroland.blobTracker.util.ElProps;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Highlighter implements Animation, TrackListener {

	private Raster r;
	private Vector<Track> tracks;
	private int camWidth = 240;
	
	public Highlighter(Raster r){
		this.r = r;
		this.tracks = new Vector<Track>();
	}

	public void initialize() {
		PGraphics c = (PGraphics)(r.getRaster());
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}

	public Raster getFrame() {
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.beginDraw();
			c.background(0);
			//System.out.println(tracks.size());
			c.noStroke();
			c.fill(255);
			for(Track t: tracks){
				// TODO iterate over blobs and draw highlighted areas
				c.rect((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width, 0, 30, c.height);
			}
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
