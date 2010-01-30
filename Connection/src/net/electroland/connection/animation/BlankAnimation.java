package net.electroland.connection.animation;

import net.electroland.connection.core.Light;

public class BlankAnimation implements Animation {
	
	public int gridx, gridy;							// grid dimensions
	public Light[] lights;								// light objects
	public byte[] buffer, pair;							// final output
	private int duration;
	
	public BlankAnimation(Light[] lights, int duration){
		this.lights = lights;
		this.duration = duration;
		gridx = 6;
		gridy = 28;
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 							// start byte
		buffer[1] = (byte)0;							// command byte
		buffer[buffer.length-1] = (byte)254; 			// end byte
	}

	public byte[] draw() {
		for(int i=0; i<lights.length; i++){				// process each light
			lights[i].setValue(0, 0);					// set everything BLACK
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];					// red
			buffer[i*2 + 3] = pair[1];					// blue
		}
		return buffer;
	}

	public int getDefaultDuration() {
		return duration;
	}

	public void start() {
		System.out.println("START: Blank Animation");
	}

	public void stop() {
		System.out.println("STOP: Blank Animation");
	}

}
