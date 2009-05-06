package net.electroland.enteractive.shows;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;

import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.enteractive.core.Model;
import net.electroland.enteractive.core.SoundManager;
import net.electroland.enteractive.core.Sprite;
import net.electroland.enteractive.core.SpriteListener;
import net.electroland.enteractive.sprites.GameOver;
import net.electroland.enteractive.sprites.Sparkler;
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
	private int introDuration;
	private Player playerA, playerB;
	private Ball ball;
	private int maxPoints = 3;
	private int maxPaddleWidth = 5;	// measured in tiles
	private PImage ballTexture, pongTitle;
	private int gameMode = 0;			// 0 = start, 1 = game play, 2 = score, 3 = end
	private boolean gameOver = false;
	private boolean playingIntro = false;
	private boolean playingScore = false;
	private boolean playingEnding = false;
	
	public Pong(Model m, Raster r, SoundManager sm, PImage ballTexture, PImage pongTitle){
		this.m = m;
		this.r = r;
		this.sm = sm;
		this.ballTexture = ballTexture;
		this.pongTitle = pongTitle;
		this.tileSize = (int)(((PGraphics)(r.getRaster())).height/11.0);
		introDuration = 5000;
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		playerA = new Player(true);
		playerB = new Player(false);
		ball = new Ball();
		startTime = System.currentTimeMillis();
		PGraphics raster = (PGraphics)(r.getRaster());
		raster.colorMode(PConstants.RGB, 255, 255, 255, 255);
	}
	

	public Raster getFrame() {
		synchronized (m){
			// presumes that you instantiated Raster with a PGraphics.
			PGraphics raster = (PGraphics)(r.getRaster());
			raster.beginDraw();
			raster.background(0);		// clear the raster
			
			switch(gameMode){
			case 0:
				// play start animation and sound
				if(!playingIntro){
					sm.createMonoSound(sm.soundProps.getProperty("pongStartSound"), 0.5f, 0.5f, 1, 1);
					playingIntro = true;
				}
				raster.image(pongTitle, 0, 0, raster.width, raster.height);
				if(System.currentTimeMillis() - startTime > introDuration){
					gameMode = 1;
					playingIntro = false;
				}
				break;
			case 1:
				gameFrame(raster);			// called when in play
				break;
			case 2:
				// play goal animation and sound, then go back to playing
				if(!playingScore){
					//sm.createMonoSound(sm.soundProps.getProperty("pongScoreSound"), 0.5f, 0.5f, 1, 1);
					Sprite s = new Sparkler(spriteIndex, r, ball.x, ball.y, sm, null, ballTexture);
					s.addListener(this);
					sprites.put(spriteIndex, s);
					spriteIndex++;
					playingScore = true;
				}
				
				raster.fill(255,0,0,255);
				for(int i=0; i<playerA.points; i++){
					raster.rect((2+(i*2))*tileSize - tileSize/2, 0 - tileSize/2, tileSize, tileSize*3);
				}
				for(int i=0; i<playerB.points; i++){
					raster.rect((15-(i*2))*tileSize - tileSize/2, 0 - tileSize/2, tileSize, tileSize*3);
				}

				if(sprites.size() == 0){
					playingScore = false;
					gameMode = 1;
					ball.reset();
				}
				break;
			case 3:
				// TODO play win/end animation and sound, then exit
				if(!playingEnding){
					sm.createMonoSound(sm.soundProps.getProperty("pongEndSound"), 0.5f, 0.5f, 1, 1);
					Sprite s = new GameOver(spriteIndex, r, ball.x, ball.y, sm);
					s.addListener(this);
					sprites.put(spriteIndex, s);
					spriteIndex++;
					playingEnding = true;
				}
				if(sprites.size() == 0){
					gameOver = true;
				}
				break;
			}
			
			// this plays celebration sprites, messages, etc
			Iterator<Sprite> spriteiter = sprites.values().iterator();
			while(spriteiter.hasNext()){
				Sprite sprite = (Sprite)spriteiter.next();
				sprite.draw();
			}
			raster.endDraw();
		}
		return r;
	}
	
	public void gameFrame(PGraphics raster){	// called when it's in game play mode (ie: ball is bouncing)
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
		//System.out.println(playerA.y1 +" "+ playerA.y2);
		if(playerA.y1 > -1 && playerA.y2 > -1){	// if start and end points established...
			if(playerA.y2 - playerA.y1 <= maxPaddleWidth){	// check if it's not too big
				playerA.paddle = true;
				//System.out.println("x:"+playerA.x*tileSize +" y:"+ playerA.y1*tileSize +" width:"+ tileSize +" height:"+ (playerA.y2-playerA.y1)*tileSize);
				raster.rect((playerA.x*tileSize)-tileSize/2, (playerA.y1*tileSize)-tileSize/2, tileSize, (1+(playerA.y2-playerA.y1))*tileSize);
			}
		}
		if(playerB.y1 > -1 && playerB.y2 > -1){	// if start and end points established...
			if(playerB.y2 - playerB.y1 <= maxPaddleWidth){	// check if it's not too big
				playerB.paddle = true;
				raster.rect((playerB.x*tileSize)-tileSize/2, (playerB.y1*tileSize)-tileSize/2, tileSize, (1+(playerB.y2-playerB.y1))*tileSize);
			}
		}
		
		// check ball position against active paddles and scoring zone
		if(ball.x-(ball.width/2) <= playerA.x*tileSize){			// in scoring zone, so check against playerA's paddle
			if(playerA.paddle){						// if paddle exists
				if(ball.y < playerA.y2*tileSize && ball.y > playerA.y1*tileSize){	// if inside paddle zone...
					ball.xvec = 0-ball.xvec;		// bounce
					//System.out.println("collision player A");
					// play paddle sound
					sm.createMonoSound(sm.soundProps.getProperty("pongPaddleSound"), 0, 0.5f, 1, 1);
				} else {
					pointScored(playerB);
				}
			} else {								// no paddle at all
				pointScored(playerB);
			}
		} else if(ball.x+(ball.width/2) >= playerB.x*tileSize){	// check against playerB's paddle
			if(playerB.paddle){						// if paddle exists
				if(ball.y < playerB.y2*tileSize && ball.y > playerB.y1*tileSize){	// if inside paddle zone...
					ball.xvec = 0-ball.xvec;		// bounce
					//System.out.println("collision player B");
					// play paddle sound
					sm.createMonoSound(sm.soundProps.getProperty("pongPaddleSound"), 1, 0.5f, 1, 1);
				} else {
					pointScored(playerA);
				}
			} else {								// no paddle at all
				pointScored(playerA);
			}
		}
		
		// check ball position against walls
		if(ball.y <= ball.width/2){					// if colliding with top...
			ball.yvec = 0-ball.yvec;				// switch directions
			sm.createMonoSound(sm.soundProps.getProperty("pongWallSound"), 0.5f, 0, 1, 1);
		} else if(ball.y >= raster.height-(ball.width/2)){		// if colliding with bottom...
			ball.yvec = 0-ball.yvec;				// switch directions
			sm.createMonoSound(sm.soundProps.getProperty("pongWallSound"), 0.5f, 1, 1, 1);
		}
		
		// draw ball
		raster.tint(255,255,255,255);
		raster.image(ballTexture, ball.x-(ball.width/2), ball.y-(ball.height/2), ball.width, ball.height);
		
		// move ball for next frame
		ball.move();
	}

	public boolean isDone() {
		return gameOver;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}
	
	public void pointScored(Player player){
		player.points++;
		if(player.points == maxPoints){
			gameMode = 3;
			startTime = System.currentTimeMillis();
		} else {
			gameMode = 2;
			startTime = System.currentTimeMillis();
		}
	}
	
	
	
	
	
	private class Player{
		public int points;		// points earned this game
		private int loc1, loc2;	// linear location in sensor grid
		private int y1, y2;		// position of feet determines bar length
		private int x;				// x and y both based on sensor grid, not raster
		private int gridWidth = 16;
		private boolean paddle = false;
		
		private Player(boolean playerA){
			if(playerA){
				x = 1;//*tileSize;
			} else {
				x = 16;//*tileSize;
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
		private int width, height;
		
		private Ball(){
			reset();
			width = height = tileSize + tileSize/2;
		}
		
		public void move(){
			x += xvec;
			y += yvec;
		}
		
		public void reset(){
			x = tileSize * 8;
			y = tileSize * 5.5f;
			xvec = (float)Math.random()*2 - 0.5f;
			if(xvec > 0){
				xvec += 0.5;
			} else {
				xvec -= 0.5;
			}
			yvec = (float)Math.random()*0.5f;
			if(yvec > 0){
				yvec += 0.25;
			} else {
				yvec -= 0.25;
			}
		}
	}

}
