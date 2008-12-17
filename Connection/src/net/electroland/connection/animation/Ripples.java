package net.electroland.connection.animation;

import net.electroland.connection.core.Light;

public class Ripples {
	
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, element;						// grid position stuff
	public Drop[] drops;
	
	public Ripples(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		drops = new Drop[5];
		drops[0] = new Drop(6, 14);
		drops[1] = new Drop(0, 28);
		drops[2] = new Drop(3, 0);
		drops[3] = new Drop(1, 6);
		drops[4] = new Drop(5, 12);
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
	}
	
	public byte[] draw(){
		for(int i=0; i<drops.length; i++){
			drops[i].expand();
		}
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	public void reset(){
		drops[0] = new Drop(6, 14);
		drops[1] = new Drop(0, 28);
		drops[2] = new Drop(3, 0);
		drops[3] = new Drop(1, 6);
		drops[4] = new Drop(5, 12);
	}
	
	
	
	
	
	public class Drop{
		public float x, y;									// normalized x and y
		public int gx, gy;									// grid x and y
		public float radius;								// radius of ripple circle (in grid units)
		public float xdiff, ydiff, hypo;					// measurements from lights to drop center
		public float age;									// age to diminish brightness
		public float threshhold;							// stroke width of circle (in grid units)
		public float brightness;
		//public int red, blue;
		
		public Drop(int _gx, int _gy){
			gx = _gx;
			gy = _gy;
			x = gx/gridx;
			y = gy/gridy;
			age = 253;
			radius = 0.1f;
			threshhold = 0.5f;
			//red = (int)Math.random()*253;
			//blue = (int)Math.random()*253;
		}
		
		public void expand(){
			radius += 0.1;									// ripple speed
			age -= 2;										// fade speed
			for(int i=0; i<lights.length; i++){
				xdiff = lights[i].x - gx;
				ydiff = lights[i].y - gy;
				hypo = (float)(Math.sqrt(xdiff*xdiff + ydiff*ydiff));
				if(radius < hypo + threshhold && radius > hypo - threshhold){	// if within stroke...
					// check for diff from radius for brightness value
					brightness = 1 - Math.abs(radius - hypo)*2;	// closer to radius, brighter it is
					lights[i].setBlue((int)(brightness*age));
					//lights[i].setValue((int)(red*brightness), (int)(blue*brightness));
					//lights[i].setValue((int)((red*brightness)*age), (int)((blue*brightness)*age));
				}
				if(age <= 0){
					// restart this ripple
					age = 253;
					radius = 0.1f;
				}
				//System.out.println(xdiff +" "+ ydiff +" "+ hypo);
			}
		}
	}

}
