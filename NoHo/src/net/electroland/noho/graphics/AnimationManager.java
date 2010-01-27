package net.electroland.noho.graphics;

import java.awt.Color;

import net.electroland.noho.core.DriveBy2010;
import net.electroland.noho.core.NoHoConfig;
import net.electroland.noho.core.TextQueue;
import net.electroland.noho.core.TrafficObserver;
import net.electroland.noho.graphics.generators.ImageFileAnimation;
import net.electroland.noho.graphics.generators.SolidColor;
import net.electroland.noho.graphics.generators.sprites.LinearMotionRecSprite;
import net.electroland.noho.graphics.generators.sprites.SpriteImageGenerator;
import net.electroland.noho.graphics.generators.sprites.TextMotionSprite;
import net.electroland.noho.graphics.generators.textAnimations.BasicText;
import net.electroland.noho.graphics.generators.textAnimations.TransitionText;
import net.electroland.noho.graphics.transitions.GenericTransition;
import net.electroland.noho.graphics.transitions.Wipe;
import net.electroland.noho.util.SimpleTimer;


/**
 * This is the class where all the state control happens.  Right now its a very crude class.  
 * @author Eitan Mendelowitz 
 * Apr 23, 2007
 */

public class AnimationManager {
	
	public enum AnimationState {INIT, TEXT, COLOR_WIPE, SPRITE, BLACK, TEXT2010}
	boolean foo = true;
	int width;
	int height;
	AnimationState state= AnimationState.INIT;
	Compositor compositor;
	TextQueue textQueue;
	TrafficObserver trafficobserver;
	
	boolean stateInited = false;
	
	ImageGenerator curForground;
	ImageGenerator curOverlay;
	ImageGenerator curBackground;

	ImageFileAnimation anim;
	ImageFileAnimation anim2;
	
	SpriteImageGenerator spriteWorld;
	
	/*
	SimpleTimer spriteTimer;
	SimpleTimer textTimer;
	// MOVED TO trafficobserver
	 * 
	 */
	
	int wipetime;
	int wipeholdtime;
	int revealtime;
	int holdtime;	
	
	// hacky:  there is no animation object for "black", we're just manipulating
	// state here.
	long blackstart = -1;
	
	boolean toggle = true;

	public AnimationManager(int w, int h, Compositor compositor, TextQueue tq, TrafficObserver trafficobserver) {
		width = w;
		height = h;
		textQueue = tq;
		this.compositor = compositor;
		this.trafficobserver = trafficobserver;
		compositor.isBackgroundEnabled(true);
		compositor.isForegroundEnabled(true);
		compositor.isOverlayEnabled(true);
		compositor.setBackgroundColor(Color.BLACK);
		anim = new ImageFileAnimation(w,h);
		anim.loadImagesAsync("./depends/wave/wave", 1, 2, ".png", 10);
		anim.playCnt(1);
		
		anim2 = new ImageFileAnimation(w,h);
		anim2.loadImagesAsync("./depends/EFFECTS/white_cells/white_cells", 0, 2, ".png", 30);
		anim2.playCnt(1);
		
		
		spriteWorld = new SpriteImageGenerator(w, h);
		
		//For Text Wipes and inter-Text reveals
		wipetime = 2000;
		//the total time the color bar is there, including reveal and hold
		wipeholdtime = 3000;
		revealtime = 2000;
		//check this before deploy, should be 10000 or more
		holdtime = NoHoConfig.PHRASETIMING;
	}

	
	public AnimationManager(int w, int h, Compositor compositor, TextQueue tq) {
		width = w;
		height = h;
		textQueue = tq;
		this.compositor = compositor;
		compositor.isBackgroundEnabled(true);
		compositor.isForegroundEnabled(true);
		compositor.isOverlayEnabled(true);
		compositor.setBackgroundColor(Color.BLACK);
				
		//For Text Wipes and inter-Text reveals
		wipetime = 2000;
		//the total time the color bar is there, including reveal and hold
		wipeholdtime = 3000;
		revealtime = 2000;
		//check this before deploy, should be 10000 or more
		holdtime = NoHoConfig.PHRASETIMING;
		anim2 = new ImageFileAnimation(w,h);
		anim2.loadImagesAsync("./depends/EFFECTS/white_cells/white_cells", 0, 2, ".png", 30);
		anim2.playCnt(1);
	}	
	
	public void nextFrame(long dt, long curTime) {

		compositor.nextFrame(dt, curTime);
		
		if (DriveBy2010.trafficEnabled){
			switch(state) {
				case INIT: {
					if(textQueue.isReady() && anim.isReady()) {
						state = AnimationState.SPRITE;
					}
				}break;
				
				case SPRITE: { // test SPRITE ImageGenerator
					if(stateInited) {
						// spriteWorld timeout, go to wipe, then to text
						if (trafficobserver.useTextMode()){
							stateInited = false;
							state = AnimationState.COLOR_WIPE;
						}
	
					} else {
						/** THESE ARE THE OLD DIAGNOSTIC SPRITES
						String tmpText = "Hi";
						TextMotionSprite s1 = new TextMotionSprite(tmpText.length()*-10,0,Color.WHITE,width,0,4000,tmpText,spriteWorld);
						TextMotionSprite s2 = new TextMotionSprite(width + tmpText.length()*10,0,Color.WHITE,0,0,4000,tmpText,spriteWorld);
						// some cute chaining syntax ;)
						spriteWorld.addListener(s1).addListener(s2);
						spriteWorld.addSprite(s1).addSprite(s2);
						*/
						
						//set revealtime to 0 to see no fade at startup
						//compositor.addForground(spriteWorld, revealtime);
						compositor.addForground(spriteWorld, revealtime);
						stateInited = true;
					}
	
				}break;
				
				case TEXT: {
					if(stateInited) {
						// so stuff in here normally until the text animation is done
						if (trafficobserver.useTextMode()) {
							// wait for this text to finish when in text mode
							if(curForground.isDone()) {
								stateInited = false;
								state = AnimationState.COLOR_WIPE;
							}
						} else {
							// immediately do the wipe to sprites
							stateInited = false;
							state = AnimationState.COLOR_WIPE;
						}
					} else {
						// if not stateInited then setup the object for the current state, and add it to Compositor FG
						
						compositor.addBackground(null, 1000);//will fade out anything what was there
						compositor.addOverlay(null, 1000);//will fade out anything what was there
						
	//					curText = new BasicText(width,height, textQueue.getNext(), 1000, 6000);
	//					SolidColor sc = new SolidColor(width, height, Color.RED, 1000); 
						anim2.reset();
	
						//Transition t = new GenericTransition(width, height, anim); // just leave out the overlay if you dont wan't it
						//Transition t = new GenericTransition(width, height, anim, new WhiteNoise(width, height));
						//Transition t = new Crossfade(width, height);
						Transition t = new GenericTransition(width, height, anim2);
	
						int lineSpeed = holdtime;
						curForground = new TransitionText(width,height, textQueue.getNext(), 1000, lineSpeed, Color.WHITE, t);
						
						//speed here controls the reveal
						compositor.addForground(curForground, revealtime);	
						stateInited = true;
					}
				}break;
				
				case COLOR_WIPE: {
					if(stateInited) {
						if(curForground.isDone()) {
							// if trafficobserver says to use text mode, continue back to text
							if (trafficobserver.useTextMode()){
								stateInited = false;
								state = AnimationState.TEXT;
							} else {
								stateInited = false;
								state = AnimationState.SPRITE;
							}			
						}
					} else {
						
						Color wipecolor = new Color((float)Math.random(),(float)Math.random(),(float)Math.random());
						int colorselector = (int)(Math.random()*100);
						if (colorselector < 25){
							wipecolor = Color.RED;
						} else if (colorselector < 50){
							wipecolor = Color.BLUE;
						} else if (colorselector < 75){
							wipecolor = Color.GREEN;
						} else {
							wipecolor = Color.WHITE;
						}
						
	//					speed here is the hold time of the color, including wipe time
						//curForground = new SolidColor(width, height, Color.GREEN, 1500);
						curForground = new SolidColor(width, height, wipecolor, wipeholdtime);
						//how fast to wipe
						compositor.addForground(curForground, wipetime);
						Transition newTrans;
						
						newTrans = new Wipe(width, height);
						
						compositor.setForgroundTransition(newTrans);
						stateInited = true;
						
					}
				} break;
			}

		}else{
			switch(state) {
				
				case INIT: {
					if(textQueue.isReady()) {
						state = AnimationState.TEXT2010;
					}
				}break;
			
				case BLACK: {
					if (curForground.isDone())
					{
						if (blackstart == -1){
							compositor.addForground(null, 0);//will fade out anything what was there
							blackstart = System.currentTimeMillis();	
						}else{
							if (System.currentTimeMillis() - blackstart > NoHoConfig.PHRASEPAUSING)
							{
								state = AnimationState.TEXT2010;
								blackstart = -1;
							}
						}
					}

				}break;
				
				case TEXT2010:{
					
					curForground = new BasicText(width,height, textQueue.getNext(), 0, NoHoConfig.PHRASETIMING);
					compositor.addForground(curForground, 0);

					state = AnimationState.BLACK;	
				}break;				
			}
		}
	}

	
	/**
	 * This is kind of hacky interface for the presence detectors
	 * to have a handle on the sprite world.
	 * 
	 * @return the currently running Sprite world.
	 */
	public SpriteImageGenerator getSpriteWorld(){
		return spriteWorld;
	}	
}
