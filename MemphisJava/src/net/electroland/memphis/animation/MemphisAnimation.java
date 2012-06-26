package net.electroland.memphis.animation;

import java.io.FileInputStream;
import java.util.Iterator;
import java.util.Properties;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;
import net.electroland.memphis.animation.sprites.Cloud;
import net.electroland.memphis.animation.sprites.Connectors;
import net.electroland.memphis.animation.sprites.DoubleWave;
import net.electroland.memphis.animation.sprites.Shooter;
import net.electroland.memphis.animation.sprites.Shooters;
import net.electroland.memphis.animation.sprites.Sprite;
import net.electroland.memphis.animation.sprites.SpriteListener;
import net.electroland.memphis.animation.sprites.Ticker;
import net.electroland.memphis.animation.sprites.Wave;
import net.electroland.memphis.core.BridgeState;
import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

public class MemphisAnimation implements Animation, SpriteListener {
	
	private boolean init = true;	// only for animations that start on launch
	private boolean isDone = false;
	private Raster raster;
	private Properties props;
	private ConcurrentHashMap<Integer,Sprite> clouds;
	private ConcurrentHashMap<Integer,Sprite> sprites;
	private int spriteIndex = 0;	// used as ID # for sprite
	private int width, height;
	private long startTime;
	private BridgeState state; // bridge state
	// shooter variables
	private PImage shooterImage;
	private float shooterLength, shooterWidth;
	private int shooterDuration;
	private int shooterFrequency;
	private int shooterBrightness;
	// ticker variables
	private PImage tickerImage;
	private float tickerLength, tickerWidth;
	private int tickerDuration;
	private int tickerOffset;
	private float[] tickerColor = new float[3];
	// wave variables
	private PImage waveImage, waveImage2;
	private int waveDuration;
	private float waveWidth;
	private float[] waveColor = new float[3];
	// cloud variables
	private PImage cloudImage;
	private int cloudDurationMin, cloudDurationMax;
	private float cloudAlpha;
	private float[] cloudColorA = new float[3];
	private float[] cloudColorB = new float[3];
	private float[] cloudColorC = new float[3];
	// connectors variables
	Connectors connectors;
	
	// BRADLEY: Modifed to pass bridge state in.  See last section of getFrame().
	public MemphisAnimation(PApplet p5, String propsFileName, BridgeState state){
		
		this.state = state;
		props = new Properties();
		try{
			props.load(new FileInputStream(propsFileName));
		} catch(Exception e){
			System.out.println(e);
		}
		clouds = new ConcurrentHashMap<Integer,Sprite>();	// for background clouds only
		sprites = new ConcurrentHashMap<Integer,Sprite>();
		width = Integer.parseInt(props.getProperty("width"));
		height = Integer.parseInt(props.getProperty("height"));
		raster = new Raster(p5.createGraphics(width, height, PConstants.P3D));
		
		// shooter variables
		shooterImage = p5.loadImage(props.getProperty("shooterImage"));
		shooterLength = Float.parseFloat(props.getProperty("shooterLength"));
		shooterWidth = Float.parseFloat(props.getProperty("shooterWidth"));
		shooterDuration = Integer.parseInt(props.getProperty("shooterDuration"));
		shooterFrequency = Integer.parseInt(props.getProperty("shooterFrequency"));
		shooterBrightness = Integer.parseInt(props.getProperty("shooterBrightness"));
		
		// ticker variables
		tickerImage = p5.loadImage(props.getProperty("tickerImage"));
		tickerLength = Float.parseFloat(props.getProperty("tickerLength"));
		tickerWidth = Float.parseFloat(props.getProperty("tickerWidth"));
		tickerDuration = Integer.parseInt(props.getProperty("tickerDuration"));
		tickerOffset = Integer.parseInt(props.getProperty("tickerOffset"));
		String[] tc = props.getProperty("tickerColor").split(",");
		tickerColor[0] = Float.parseFloat(tc[0]);
		tickerColor[1] = Float.parseFloat(tc[1]);
		tickerColor[2] = Float.parseFloat(tc[2]);
		
		// wave variables
		waveImage = p5.loadImage(props.getProperty("waveImage"));
		waveImage2 = p5.loadImage(props.getProperty("waveImage2"));
		waveDuration = Integer.parseInt(props.getProperty("waveDuration"));
		waveWidth = Float.parseFloat(props.getProperty("waveWidth"));
		String[] wc = props.getProperty("waveColor").split(",");
		waveColor[0] = Float.parseFloat(wc[0]);
		waveColor[1] = Float.parseFloat(wc[1]);
		waveColor[2] = Float.parseFloat(wc[2]);
		
		// cloud variables
		cloudImage = p5.loadImage(props.getProperty("cloudImage"));
		cloudDurationMin = Integer.parseInt(props.getProperty("cloudDurationMin"));
		cloudDurationMax = Integer.parseInt(props.getProperty("cloudDurationMax"));
		cloudAlpha = Float.parseFloat(props.getProperty("cloudAlpha"));
		String[] cc = props.getProperty("cloudColorA").split(",");
		cloudColorA[0] = Float.parseFloat(cc[0]);
		cloudColorA[1] = Float.parseFloat(cc[1]);
		cloudColorA[2] = Float.parseFloat(cc[2]);
		cc = props.getProperty("cloudColorB").split(",");
		cloudColorB[0] = Float.parseFloat(cc[0]);
		cloudColorB[1] = Float.parseFloat(cc[1]);
		cloudColorB[2] = Float.parseFloat(cc[2]);
		cc = props.getProperty("cloudColorC").split(",");
		cloudColorC[0] = Float.parseFloat(cc[0]);
		cloudColorC[1] = Float.parseFloat(cc[1]);
		cloudColorC[2] = Float.parseFloat(cc[2]);
	}

	public Raster getFrame() {
		PGraphics c = (PGraphics)raster.getRaster();
		
		if(init){
			init = false;
			// create cloud sprites that will run constantly in the background
			int cloudDuration = (int)((float)Math.random() * (cloudDurationMax - cloudDurationMin)) + cloudDurationMin;
			Cloud cloudA = new Cloud(spriteIndex, raster, 0 - (float)Math.random()*(cloudImage.width/2), 0, cloudImage, cloudDuration);
			//Cloud cloudA = new Cloud(spriteIndex, raster, 0, 0, cloudImage, cloudDuration);
			cloudA.setColor(cloudColorA[0], cloudColorA[1], cloudColorA[2], cloudAlpha);
			clouds.put(spriteIndex, cloudA);
			spriteIndex++;
			//cloudDuration = (int)((float)Math.random() * (cloudDurationMax - cloudDurationMin)) + cloudDurationMin;
			//Cloud cloudB = new Cloud(spriteIndex, raster, 0 - (float)Math.random()*(cloudImage.width/2), 0, cloudImage, cloudDuration);
			//cloudB.setColor(cloudColorB[0], cloudColorB[1], cloudColorB[2], cloudAlpha);
			//clouds.put(spriteIndex, cloudB);
			//spriteIndex++;
			cloudDuration = (int)((float)Math.random() * (cloudDurationMax - cloudDurationMin)) + cloudDurationMin;
			Cloud cloudC = new Cloud(spriteIndex, raster, 0 - (float)Math.random()*(cloudImage.width/2), 0, cloudImage, cloudDuration);
			cloudC.setColor(cloudColorC[0], cloudColorC[1], cloudColorC[2], cloudAlpha);
			clouds.put(spriteIndex, cloudC);
			spriteIndex++;
			
			/*
			// TODO: uncomment this to have connectors drawn between recently triggered bays
			connectors = new Connectors(spriteIndex, raster, 0, 0, state);
			sprites.put(spriteIndex, connectors);
			spriteIndex++;
			*/
			
			startTime = System.currentTimeMillis();	// timer controls frequency of shooters emitted in background
		}
		
		/*
		if(System.currentTimeMillis() - startTime > shooterFrequency){
			float ypos = (float)Math.floor((Math.random() * 4)) * c.height/4;
			boolean flip = false;
			if(Math.random() > 0.5){
				flip = true;
			}
			Shooter shooter = new Shooter(spriteIndex, raster, shooterImage, 0, ypos, shooterLength, shooterWidth, shooterDuration, flip);
			if(flip){	// blue hues
				shooter.setColor(0.0f, (float)Math.random() * shooterBrightness, shooterBrightness);
			} else {	// green hues
				shooter.setColor(0.0f, (float)Math.random() * shooterBrightness, shooterBrightness);
				//shooter.setColor(0.0f, shooterBrightness, (float)Math.random() * shooterBrightness);
			}
			shooter.addListener(this);
			sprites.put(spriteIndex, shooter);
			spriteIndex++;
			startTime = System.currentTimeMillis();
		}
		*/
		
		// render current state of animation
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		c.beginDraw();
		c.background(0);
		// draw background clouds
		Iterator<Sprite> cloudIter = clouds.values().iterator();
		while(cloudIter.hasNext()){
			Sprite sprite = (Sprite)cloudIter.next();
			sprite.draw();
		}
		// draw sprites
		Iterator<Sprite> iter = sprites.values().iterator();
		while(iter.hasNext()){
			Sprite sprite = (Sprite)iter.next();
			sprite.draw();
		}
		c.endDraw();
		
		/** BRADLEY: Example case for polling for which sensors to process */
		for (int i = 0; i < state.getSize(); i++){
			if (state.requiresNewSprite(i)){ // see if any sensor is ready for action.
				
				// start a new sprite for bridge at position i here.
				float xpos = ((width/27) * i) + tickerOffset;
				Ticker ticker = new Ticker(spriteIndex, raster, xpos, 0.0f, tickerImage, tickerWidth, tickerLength, tickerDuration, false);
				ticker.setColor(tickerColor[0], tickerColor[1], tickerColor[2]);
				ticker.addListener(this);
				sprites.put(spriteIndex, ticker);
				spriteIndex++;
				
				if(i == 0){
					// if first sensor, send a big sprite down the whole length of the bridge
					Wave wave = new Wave(spriteIndex, raster, xpos, 0.0f, waveImage, waveWidth, height, waveDuration, false);
					wave.setColor(waveColor[0], waveColor[1], waveColor[2]);
					wave.addListener(this);
					sprites.put(spriteIndex, wave);
					spriteIndex++;
				} else if(i == 26){
					// if last sensor, send a big sprite down the whole length of the bridge
					Wave wave = new Wave(spriteIndex, raster, xpos, 0.0f, waveImage, waveWidth, height, waveDuration, true);
					wave.setColor(waveColor[0], waveColor[1], waveColor[2]);
					wave.addListener(this);
					sprites.put(spriteIndex, wave);
					spriteIndex++;
				} else if(i == 13){ // changed to 13 in 2012 to accurately reflect center of bridge
					/*
					// TODO: uncomment this to have shooters emit from a person when they continually trigger a sensor.
					// start a new sprite for shooters at position i
					Shooters shooters = new Shooters(spriteIndex, raster, xpos, 0, shooterImage, shooterLength, shooterWidth, shooterDuration, shooterFrequency, shooterBrightness, state, i);
					shooters.addListener(this);
					sprites.put(spriteIndex, shooters);
					spriteIndex++;
					*/
					DoubleWave doublewave = new DoubleWave(spriteIndex, raster, xpos, 0, waveImage, waveImage2, waveWidth, height, waveDuration, 6000);
					doublewave.addListener(this);
					sprites.put(spriteIndex, doublewave);
					spriteIndex++;
				} else if(i == 7 || i == 19){ // new in 2012 create extra red waves
					DoubleWave doublewave = new DoubleWave(spriteIndex, raster, xpos, 0, waveImage, waveImage2, waveWidth, height, waveDuration, 6000);
					doublewave.addListener(this);
					sprites.put(spriteIndex, doublewave);
					spriteIndex++;
				} 
				
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
