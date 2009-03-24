package net.electroland.enteractive.sprites;

import processing.core.PConstants;
import processing.core.PGraphics;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.lighting.detector.animation.Raster;

public class TickerBox extends Sprite {
	
	private int duration, tickerSpeed, brightness, state;
	private long startTime, tickerTime;

	public TickerBox(int id, Raster raster, float x, float y, SoundManager sm, int duration) {
		super(id, raster, x, y, sm);
		this.duration = duration;
		tickerSpeed = 30;
		startTime = System.currentTimeMillis();
		tickerTime = System.currentTimeMillis();
		brightness = 255;
		state = 0;
	}

	@Override
	public void draw() {
		brightness = 255 - (int)(((System.currentTimeMillis() - startTime) / (float)duration) * 255);
		if(brightness <= 0){
			die();
		}
		
		if((System.currentTimeMillis() - tickerTime) >= tickerSpeed){
			if(state < 7){
				state++;
			} else {
				state = 0;
			}
			tickerTime = System.currentTimeMillis();
		}
		
		if(raster.isProcessing()){
			PGraphics c = (PGraphics)canvas;
			c.pushMatrix();
			c.rectMode(PConstants.CENTER);				// centered at sprite's X/Y position
			if(state == 0){
				c.fill(brightness,0,0);
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*6,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left 
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*4,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square 
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*1,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right 
				c.fill(0,0,0);
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 1){
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill(0,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill(brightness,0,0);
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 2){
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill(0,0,0);
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill(brightness,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x, y-tileSize, tileSize, tileSize);			// top				
			} else if(state == 3){
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill(0,0,0);
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill(brightness,0,0);
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 4){
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill(0,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill(brightness,0,0);
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 5){
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill(0,0,0);
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill(brightness,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 6){
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill(0,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill(brightness,0,0);
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			} else if(state == 7){
				c.fill(0,0,0);
				c.rect(x-tileSize, y-tileSize, tileSize, tileSize);	// top left 
				c.fill(brightness,0,0);
				c.rect(x-tileSize, y, tileSize, tileSize);			// left
				c.fill((brightness/7.0f)*6,0,0); 
				c.rect(x-tileSize, y+tileSize, tileSize, tileSize);	// bottom left
				c.fill((brightness/7.0f)*5,0,0);
				c.rect(x, y+tileSize, tileSize, tileSize);			// bottom square
				c.fill((brightness/7.0f)*4,0,0); 
				c.rect(x+tileSize, y+tileSize, tileSize, tileSize);	// bottom right
				c.fill((brightness/7.0f)*3,0,0);
				c.rect(x+tileSize, y, tileSize, tileSize);			// right
				c.fill((brightness/7.0f)*2,0,0);
				c.rect(x+tileSize, y-tileSize, tileSize, tileSize);	// top right
				c.fill((brightness/7.0f)*1,0,0); 
				c.rect(x, y-tileSize, tileSize, tileSize);			// top
			}
			c.popMatrix();
		}
	}

}
