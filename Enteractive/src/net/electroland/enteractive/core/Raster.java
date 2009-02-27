package net.electroland.enteractive.core;

import java.awt.Graphics;
import java.awt.image.BufferedImage;
import processing.core.PGraphics;

public class Raster {
	
	private Graphics graphic;
	private PGraphics pgraphic;
	
	public Raster(Graphics graphic){
		this.graphic = graphic;
	}
	
	public Raster(PGraphics pgraphic){
		this.pgraphic = pgraphic;
	}
	
	public BufferedImage getRaster(){	// output for ArtNetBroadcaster
		if(graphic == null){
			// TODO: Need to convert from PGraphics to BufferedImage
			return null;
		} else {
			// TODO: Need to convert from Graphics to BufferedImage
			return null;
		}
	}
}
