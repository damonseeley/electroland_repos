package net.electroland.laface.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.laface.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class Bars extends Sprite {

	static private final int GRIDLENGTH = 174;
	private double Y[] = new double[GRIDLENGTH];  // numerical grid
	private int brightness, alpha;
	private int xoffs, yoffs, width, height;
	static private double xscale, yscale;
	
	public Bars(int id, Raster raster, float x, float y) {
		super(id, raster, x, y);
		brightness = 255;
		alpha = 255;
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			width = c.width;
			height = c.height;
			xoffs = 0;
			yoffs = height/2 + height/4;
			xscale = c.width/(float)(GRIDLENGTH-1);
			yscale = c.height/3;
		}
		for(int i=0; i<GRIDLENGTH; i++){
			Y[i] = 0;
		}
	}

	@Override
	public void draw(Raster r) {
		//System.out.println("bars drawing");
		if(r.isProcessing()){
			PGraphics c = (PGraphics)(r.getRaster());
			c.noStroke();
			c.fill(brightness,brightness,brightness,alpha);
			//c.beginShape();
			//int lowest = c.height;	// lowest point (highest value) in bars
			int px, py, x, y;
			px = xoffs;
			py = (int)(Y[0]*yscale + yoffs);
			//c.vertex(px,py);
			for(int i=1; i<GRIDLENGTH; i++) {
				x = (int)(i*xscale) + xoffs;
				y = (int)(Y[i]*yscale + yoffs);
				c.rect(px, py+((y-py)/2), x-px, c.height-(py+((y-py)/2)));	// vertical bar for each point
				//System.out.println(x+" "+y);
				//c.vertex(x,y);
				//if(y > lowest){
				//	lowest = y;
				//}
				px = x;
				py = y;
			}
			//c.vertex(c.width,lowest);
			//c.vertex(0,lowest);
			//c.endShape(PConstants.CLOSE);
		}
		// reset the Y locations, since swells are updated on next frame draw
		for(int i=0; i<GRIDLENGTH; i++){
			Y[i] = 0;
		}
	}
	
	public void swell(float x, float amplitude, int wavelength){
		int xpos = (int)(x*GRIDLENGTH);		// position in array
		Y[xpos] -= amplitude;					// add intensity to center bar
		for(int i=1; i<=wavelength/2; i++){
			if(xpos+i < GRIDLENGTH){
				Y[xpos+i] -= amplitude;			// straight top
			} else if (xpos-i >= 0){
				Y[xpos-i] -= amplitude;			// straight top
			}
		}
	}
	
	
	
	
	
	
	// ALLOW CONTROL PANEL TO SET PROPERTIES
	
	public void setAlpha(int alpha){
		if(alpha >= 0 && alpha <= 255){
			this.alpha = alpha;
		}
	}
	
	public void setBrightness(int brightness){
		if(brightness >= 0 && brightness <= 255){
			this.brightness = brightness;
		}
	}
	
	public void setYoffset(double yoffset){
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)(raster.getRaster());
			yoffs = (int)(yoffset*c.height);
		}
	}

}
