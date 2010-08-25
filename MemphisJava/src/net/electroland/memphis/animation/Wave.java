package net.electroland.memphis.animation;

import java.io.FileInputStream;
import java.util.Properties;

import processing.core.PApplet;
import processing.core.PConstants;
import processing.core.PGraphics;
import processing.core.PImage;

import net.electroland.lighting.detector.animation.Animation;
import net.electroland.lighting.detector.animation.Raster;

public class Wave implements Animation{
	
	private boolean isDone = false;
	private Raster raster;
	private Properties props;
	private PImage gradientImage;
	private int width, height;
	private int waveDuration;
	private int waveWidth;
	private long startTime;
	
	public Wave(PApplet p5, String propsFileName){
		props = new Properties();
		try{
			props.load(new FileInputStream(propsFileName));
		} catch(Exception e){
			System.out.println(e);
		}
		width = Integer.parseInt(props.getProperty("width"));
		height = Integer.parseInt(props.getProperty("height"));
		waveDuration = Integer.parseInt(props.getProperty("waveDuration"));
		waveWidth = Integer.parseInt(props.getProperty("waveWidth"));
		raster = new Raster(p5.createGraphics(width, height, PConstants.P3D));
		gradientImage = p5.loadImage(props.getProperty("image"));
		startTime = System.currentTimeMillis();
	}

	public Raster getFrame() {
		// calculate X position of wave graphic
		float xpos = (((System.currentTimeMillis() - startTime) / (float)waveDuration) * (width + gradientImage.width*2)) - waveWidth;
		if(System.currentTimeMillis() - startTime > waveDuration){
			startTime = System.currentTimeMillis();
		}
		//System.out.println(xpos);
		
		// render current state of animation
		PGraphics c = (PGraphics)raster.getRaster();
		c.colorMode(PConstants.RGB, 255, 255, 255, 255);
		c.beginDraw();
		c.background(0);
		c.image(gradientImage, xpos, 0, waveWidth, height);
		c.endDraw();
		return raster;
	}

	public void init(Properties props) {
		this.props = props;
	}

	public boolean isDone() {
		return isDone;
	}

	public void stop(){
		isDone = true;
	}

}
