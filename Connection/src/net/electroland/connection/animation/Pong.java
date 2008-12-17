package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class Pong {
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public Ball ball;										// ball in play
	public int gridx, gridy, element, row;					// grid position stuff
	public int xpos, ypos;
	
	public Pong(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		ball = new Ball();									// moving ball
	}
	
	public byte[] draw(Person[] peoplelist){
		for(int i=0; i<peoplelist.length; i++){
			try{
				int[] loc = peoplelist[i].getIntLoc();		// person location
				xpos = loc[0];
				ypos = loc[1];
				element = xpos + ypos*gridx;
				if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
					//lights[element].setBlue(253);				// make light active
					for(int n=0; n<gridx; n++){				// draw row above person
						lights[(ypos*gridx)+n].setBlue(253);
					}
					if(ball.gy == ypos){
						// play sound above person
						ball.bounce();
					}
				}
				ball.move();
			}catch (NullPointerException e){
				e.printStackTrace();
			}
		}
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	public void start() {
		// start the global sound
	}
	
	public void stop() {
		// stop the global sound
	}
	
	
	
	
	public class Ball{
		public float y;
		public float yvec;
		public int gy, gx, element;
		public int red, blue;
		public int soundID;
		public String soundFile;
		public int bouncedelay;
		public Explosion explosion;
		
		public Ball(){
			y = 0;
			yvec = (float)(0.005f*Math.random()) + 0.002f;
			gx = (int)(Math.random()*5.99);
			gy = 0;
			element = gy*gridx + gx;
			red = 255;
			blue = 255;
			bouncedelay = 0;
		}
		
		public void bounce(){
			if(bouncedelay == 0){
				yvec = 0-yvec;
				bouncedelay = 15;
				blue -= 15;
				if(blue <= 0){
					explosion = new Explosion(0, gx, gy);
				}
			}
		}
		
		public void move(){
			if(blue <= 0 && !explosion.done){
				explosion.expand();
			} else if(blue <= 0 && explosion.done){
				y = 0;
				yvec = (float)(0.005f*Math.random()) + 0.002f;
				gx = (int)(Math.random()*5.99);
				element = gy*gridx + gx;
				blue = 255;
				bouncedelay = 0;
			} else {
				y += yvec;
				if(bouncedelay > 0){
					bouncedelay--;
				}
				if(y >= 1 || y <= 0){
					bounce();
				}
				gy = (int)(y*27.99);
				element = gy*gridx + gx;
				if(element < lights.length && element >= 0){
					lights[element].setValue(red, blue);				// make light active
				}
			}
		}
	}
	
	
	
	
	public class Explosion{
		public float x, y;									// normalized x and y
		public int gx, gy;									// grid x and y
		public int id;
		public float radius;								// radius of explosive circle (in grid units)
		public float xdiff, ydiff, hypo;					// measurements from lights to impact center
		public float age;									// age to diminish brightness
		public float threshhold;							// stroke width of circle (in grid units)
		public float brightness;
		public boolean done;
		public String soundFile;
		public int[] soundloc;
		public int soundID;
		public int gain = 1;
		
		public Explosion(int _id, int _gx, int _gy){
			id = _id;
			gx = _gx;
			gy = _gy;
			x = gx/gridx;
			y = gy/gridy;
			age = 253;
			radius = 0.1f;
			threshhold = 0.5f;
			done = false;
			soundFile = ConnectionMain.properties.get("soundExplosion");
//			try{
//				soundloc = ConnectionMain.soundController.getNearestSpeaker(gx,gy);
//				soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
//				ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundloc[0]+" "+soundloc[1]+" 0 "+gain);
//			} catch(NullPointerException e){
//				System.err.println("sound output error: "+e);
//			}
		}
		
		public void expand(){
			radius += 0.2;									// explosion outward speed
			age -= 10;										// fade speed
			for(int i=0; i<lights.length; i++){
				xdiff = lights[i].x - gx;
				ydiff = lights[i].y - gy;
				hypo = (float)(Math.sqrt(xdiff*xdiff + ydiff*ydiff));
				if(radius < hypo + threshhold && radius > hypo - threshhold){	// if within stroke...
					brightness = 1;
					//brightness = 1 - Math.abs(radius - hypo)*2;	// closer to radius, brighter it is
					lights[i].setRed((int)(brightness*age));
				}
				if(age <= 0){
					done = true;
				}
			}
		}
	}
	
}
