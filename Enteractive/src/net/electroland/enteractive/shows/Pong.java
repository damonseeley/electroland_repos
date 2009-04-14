package net.electroland.enteractive.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;

import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Pong implements Animation, SpriteListener {
	
	private Model m;
	private Raster r;
	private SoundManager sm;
	private int tileSize;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;
	private long startTime;
	private int duration = 10000;		// milliseconds
	private Player playerA, playerB;
	private Ball ball;
	private int maxPoints = 3;
	private int maxPaddleWidth = 5;	// measured in tiles
	
	public Pong(Model m, Raster r, SoundManager sm){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		playerA = new Player(true);
		playerB = new Player(false);
	}

	public void initialize() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
		raster.rectMode(PConstants.CENTER);
		startTime = System.currentTimeMillis();
	}

	public Raster getFrame() {
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			
			boolean[] sensors = m.getSensors();
			// check all of player A's sensors
			playerA.resetLoc();
			for(int i=0; i<sensors.length; i+=16){
				if(sensors[i]){						// if it's on...
					if(playerA.loc1 < 0){			// first loc not registered yet...
						playerA.setLoc1(i);
					} else if(playerA.loc2 < 0) {	// second loc not registered yet...
						playerA.setLoc2(i);
					}
				}
			}
			
			// check all of player B's sensors
			playerB.resetLoc();
			for(int i=15; i<sensors.length; i+=16){
				if(sensors[i]){						// if it's on...
					if(playerB.loc1 < 0){			// first loc not registered yet...
						playerB.setLoc1(i);
					} else if(playerB.loc2 < 0) {	// second loc not registered yet...
						playerB.setLoc2(i);
					}
				}
			}
			
			raster.fill(255,0,0);
			// draw bars
			if(playerA.y1 > -1 && playerA.y2 > -1){	// if start and end points established...
				if(playerA.y2 - playerA.y1 <= maxPaddleWidth){	// check if it's not too big
					playerA.paddle = true;
					raster.rect(playerA.x*tileSize, playerA.y1*tileSize, tileSize, (playerA.y2-playerA.y1)*tileSize);
				}
			}
			if(playerB.y1 > -1 && playerB.y2 > -1){	// if start and end points established...
				if(playerB.y2 - playerB.y1 <= maxPaddleWidth){	// check if it's not too big
					playerB.paddle = true;
					raster.rect(playerB.x*tileSize, playerB.y1*tileSize, tileSize, (playerB.y2-playerB.y1)*tileSize);
				}
			}
			
			// check ball position against active paddles, walls, scoring zone
			if(ball.x <= playerA.x*tileSize){			// in scoring zone, so check against playerA's paddle
				if(playerA.paddle){						// if paddle exists
					if(ball.x < playerA.y2*tileSize && ball.x > playerA.y1*tileSize){	// if inside paddle zone...
						ball.xvec = 0-ball.xvec;		// bounce
					} else {
						// TODO point to playerB
					}
				} else {								// no paddle at all
					// TODO point to playerB
				}
			} else if(ball.x >= playerB.x*tileSize){	// check against playerB's paddle
				if(playerB.paddle){						// if paddle exists
					if(ball.x < playerB.y2*tileSize && ball.x > playerB.y1*tileSize){	// if inside paddle zone...
						ball.xvec = 0-ball.xvec;		// bounce
					} else {
						// TODO point to playerA
					}
				} else {								// no paddle at all
					// TODO point to playerA
				}
			}
			
			// this may not be necessary
			//Iterator<Sprite> spriteiter = sprites.values().iterator();
			//while(spriteiter.hasNext()){
			//	Sprite sprite = (Sprite)spriteiter.next();
			//	sprite.draw();
			//}
			raster.endDraw();
		}
		return r;
	}

	public void cleanUp() {
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.beginDraw();
		raster.background(0);
		raster.endDraw();
	}

	public boolean isDone() {
		if(playerA.points >= maxPoints || playerB.points >= maxPoints){
			return true;
		}
		return false;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}
	
	
	
	
	
	private class Player{
		private int points;		// points earned this game
		private int loc1, loc2;	// linear location in sensor grid
		private int y1, y2;		// position of feet determines bar length
		private int x;				// x and y both based on sensor grid, not raster
		private int gridWidth = 16;
		private int gridHeight = 11;
		private boolean paddle = false;
		
		private Player(boolean playerA){
			if(playerA){
				x = 1*tileSize;
			} else {
				x = 17*tileSize;
			}
		}
		
		public void resetLoc(){
			loc1 = loc2 = -1;
			y1 = y2 = -1;
			paddle = false;
		}
		
		public void setLoc1(int loc){
			loc1 = loc;
			y1 = loc / gridWidth;
		}
		
		public void setLoc2(int loc){
			loc2 = loc;
			y2 = loc / gridWidth;
		}
	}
	
	
	
	
	
	private class Ball{
		private float x, y, xvec, yvec;	// all based on raster, not sensor grid
		
		private Ball(){
			
		}
	}

}
