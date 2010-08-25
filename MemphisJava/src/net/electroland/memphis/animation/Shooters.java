package net.electroland.memphis.animation;

import java.io.FileInputStream;
import java.util.Iterator;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.memphis.animation.sprites.Shooter;
import net.electroland.memphis.animation.sprites.Sprite;
import net.electroland.memphis.animation.sprites.SpriteListener;
import net.electroland.memphis.core.BridgeState;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

public class Shooters implements Animation, SpriteListener {
	

	private boolean isDone = false;
	private Raster raster;
	private Properties props;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;	// used as ID # for sprite
	private int width, height;
	private PImage shooterImage;
	private float shooterLength, shooterWidth;
	private int shooterDuration;
	private int shooterFrequency;
	private long startTime;
	private BridgeState state; // bridge state
	
	// BRADLEY: Modifed to pass bridge state in.  See last section of getFrame().
	public Shooters(PApplet p5, String propsFileName, BridgeState state){
		
		this.state = state;
		props = new Properties();
		try{
			props.load(new FileInputStream(propsFileName));
		} catch(Exception e){
			System.out.println(e);
		}
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		width = Integer.parseInt(props.getProperty("width"));
		height = Integer.parseInt(props.getProperty("height"));
		shooterLength = Float.parseFloat(props.getProperty("shooterLength"));
		shooterWidth = Float.parseFloat(props.getProperty("shooterWidth"));
		shooterDuration = Integer.parseInt(props.getProperty("shooterDuration"));
		shooterFrequency = Integer.parseInt(props.getProperty("shooterFrequency"));
		raster = new Raster(p5.createGraphics(width, height, PConstants.P3D));
		shooterImage = p5.loadImage(props.getProperty("image"));
		startTime = System.currentTimeMillis();
	}

	public Raster getFrame() {
		if(System.currentTimeMillis() - startTime > shooterFrequency){
			float ypos = (float)Math.floor((Math.random() * 4) * 9);
			boolean flip = false;
			if(Math.random() > 0.5){
				flip = true;
			}
			Shooter shooter = new Shooter(spriteIndex, raster, shooterImage, 0, ypos, shooterLength, shooterWidth, shooterDuration, flip);
			if(flip){	// blue hues
				shooter.setColor(0.0f, (float)Math.random() * 255, 255.0f);
			} else {	// green hues
				shooter.setColor(0.0f, 255.0f, (float)Math.random() * 255);
			}
			shooter.addListener(this);
			sprites.put(spriteIndex, shooter);
			spriteIndex++;
			startTime = System.currentTimeMillis();
		}
		
		// render current state of animation
		PGraphics c = (PGraphics)raster.getRaster();
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		c.beginDraw();
		c.background(0);
		// draw shooters
		Iterator<Sprite> iter = sprites.values().iterator();
		while(iter.hasNext()){
			Sprite sprite = (Sprite)iter.next();
			sprite.draw();
		}
		c.endDraw();
		
		/** BRADLEY: Example case for polling for which sensors to process */
		for (int i = 0; i < state.getSize(); i++){
			if (state.requiresNewSprite(i)){ // see if any sensor is ready for action.
				
				// start a new srpite for bridge at position i here.

				state.spriteStarted(i); // let the bridge state know you started some action for that sensor.
			}
		}
		
		/*******************************/

		return raster;
	}
	long processTime;
	
	public void init(Properties props) {
		this.props = props;
	}

	public boolean isDone() {
		return isDone;
	}

	public void spriteComplete(Sprite sprite) {
		sprites.remove(sprite.getID());
	}

}
