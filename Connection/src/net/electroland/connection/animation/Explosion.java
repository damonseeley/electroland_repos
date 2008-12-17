package net.electroland.connection.animation;

import java.util.concurrent.ConcurrentHashMap;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Explosion{
	
	public Light[] lights;
	public ConcurrentHashMap<Integer, Blast> blasts;
	public float x, y;									// normalized x and y
	public int gridx, gridy;
	public int gx, gy;									// grid x and y
	public int id;
	public float radius;								// radius of explosive circle (in grid units)
	public float xdiff, ydiff, hypo;					// measurements from lights to impact center
	public float age;									// age to diminish brightness
	public float threshhold;							// stroke width of circle (in grid units)
	public float brightness, red, blue;
	public int fadespeed;								// rate of diminish in brightness increments
	public boolean done;
	public String soundFile;
	
	public Explosion(Light[] lights, int id, int gx, int gy){
		this.lights = lights;
		this.id = id;
		this.gx = gx;
		this.gy = gy;
		this.red = 1;	// normalized color value
		this.blue = 0;
		gridx = 6;
		gridy = 28;
		x = gx/gridx;
		y = gy/gridy;
		age = 253;
		radius = 0.1f;
		threshhold = 0.5f;
		fadespeed = 10;
		done = false;
		soundFile = ConnectionMain.properties.get("soundExplosion");
		ConnectionMain.soundController.playSimpleSound(soundFile, gx, gy, 0.8f, "explosion");
	}
	
	public Explosion(Light[] lights, int id, int gx, int gy, int red, int blue, int fadespeed){
		this.lights = lights;
		this.id = id;
		this.gx = gx;
		this.gy = gy;
		this.red = (red/253.0f);	// normalized color value
		this.blue = (blue/253.0f);
		this.fadespeed = fadespeed;
		gridx = 6;
		gridy = 28;
		x = gx/gridx;
		y = gy/gridy;
		age = 253;
		radius = 0.1f;
		threshhold = 0.5f;
		done = false;
		soundFile = ConnectionMain.properties.get("soundExplosion");
		ConnectionMain.soundController.playSimpleSound(soundFile, gx, gy, 0.8f, "explosion");
	}
	
	public void expand(){
		radius += 0.2;									// explosion outward speed
		age -= fadespeed;								// fade speed
		for(int i=0; i<lights.length; i++){
			xdiff = lights[i].x - gx;
			ydiff = lights[i].y - gy;
			hypo = (float)(Math.sqrt(xdiff*xdiff + ydiff*ydiff));
			if(radius < hypo + threshhold && radius > hypo - threshhold){	// if within stroke...
				brightness = 1;
				//brightness = 1 - Math.abs(radius - hypo)*2;	// closer to radius, brighter it is
				//lights[i].setRed((int)(brightness*age));		// brightness value no longer being used
				lights[i].addValue((int)(red*age),(int)(blue*age));
			}
			if(age <= 0){
				done = true;
			}
		}
	}
}
