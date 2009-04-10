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
	private int camWidth = 240;
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
			c.background(0);
			//System.out.println(tracks.size());
			c.noStroke();
			c.fill(255);
			for(Track t: tracks){
				// TODO iterate over blobs and draw highlighted areas
				//c.rect((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width, 0, 30, c.height);
				c.image(texture, ((t.x/Integer.parseInt(ElProps.THE_PROPS.get("srcWidth").toString()))*c.width) - 25, 0, 50, c.height);
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
