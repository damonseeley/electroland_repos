package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Biggest implements Animation{
	
	public int gridx, gridy;							// grid dimensions
	public byte[] buffer, pair;						// final output
	public Light[] lights;								// light objects
	private int fade = 0;
	private boolean hold = true;
	private int holdtimer, holdlength, fadespeed;
	private int red, blue;
	private int soundID;
	private String soundFile;
	private int duration;

	public Biggest(Light[] lights, int holdlength, int fadespeed, int duration){
		this.lights = lights;
		this.holdlength = holdlength;
		this.fadespeed = fadespeed;
		this.duration = duration;
		gridx = 6;
		gridy = 28;
		red = 255;
		blue = 0;
		buffer = new byte[(gridy*gridx)*2 + 3];		// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 							// start byte
		buffer[1] = (byte)0;							// command byte
		buffer[buffer.length-1] = (byte)254; 			// end byte
		soundFile = ConnectionMain.properties.get("soundBiggestGlobal");
		soundID = -1;
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
	public byte[] draw(){
		if(hold){
			if(holdtimer < holdlength){
				holdtimer++;
			} else {
				holdtimer = 0;
				hold = false;
			}
		} else {
			if(fade == 0){			// fade from red to purple
				if(blue < 255){
					blue += fadespeed;
				} else if(blue >= 255){
					hold = true;
					fade = 1;
				}
			} else if(fade == 1){	// fade from purple to blue
				if(red > 0){
					red -= fadespeed;
				} else if(red <= 0){
					hold = true;
					fade = 2;
				}
			} else if(fade == 2){	// fade from blue to purple
				if(red < 255){
					red += fadespeed;
				} else if(red >= 255){
					hold = true;
					fade = 3;
				}
			} else if(fade == 3){
				if(blue > 0){
					blue -= fadespeed;
				} else if(blue <= 0){
					hold = true;
					fade = 0;
				}
			}
		}

		if(red > 255){			// protect from going out of range
			red = 255;
		} else if(red < 0){
			red = 0;
		}
		
		if(blue > 255){			// protect from going out of range
			blue = 255;
		} else if(blue < 0){
			blue = 0;
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			lights[i].setValue(red, blue);
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Giant Color Shift");
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,false,1,10000,"biggest");
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Giant Color Shift " + soundID);
		ConnectionMain.soundController.killSound(soundID);
	}
	
}
