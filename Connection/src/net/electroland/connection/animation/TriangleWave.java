package net.electroland.connection.animation;

import net.electroland.connection.core.Light;

public class TriangleWave {
	
	public Light[] lights;
	public byte[] buffer;
	public byte[] pair;
	public boolean fadein = true;
	public int value = 0;
	public int fadespeed = 5;
	
	public TriangleWave(Light[] _lights){
		lights = _lights;
		buffer = new byte[(28*6)*2 + 3];					// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
	}
	
	public byte[] draw(){
		for(int i=0; i<lights.length; i++){
			lights[i].setRed(value);
		}
		if(fadein && value < 255){
			value += fadespeed;
		} else if (fadein && value >= 255){
			value = 255;
			fadein = false;
		} else if(fadein == false && value > 0){
			value -= fadespeed;
		} else if(fadein == false && value <= 0){
			fadein = true;
			value = 0;
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
}
