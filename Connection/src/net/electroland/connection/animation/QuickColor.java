package net.electroland.connection.animation;

import net.electroland.connection.core.Light;
import net.electroland.connection.core.ConnectionMain;

public class QuickColor implements Animation{
	
	public int gridx, gridy;							// grid dimensions
	public byte[] buffer, pair;						// final output
	public Light[] lights;								// light objects
	public int soundID;
	public String soundFile;
	private int duration;
	private boolean hold = false;
	private int holdtimer, holdlength, fadeinspeed, fadeoutspeed;
	private int red, blue, targetred, targetblue;
	private boolean fadein, fadeout;
	
	public QuickColor(int duration, int red, int blue, String soundFile){
		this.duration = duration;
		this.targetred = red;
		this.targetblue = blue;
		this.soundFile = soundFile;
		gridx = 6;
		gridy = 28;
		fadein = true;
		fadeinspeed = 17; 	// 0.5 seconds,  color intensity shift per frame
		fadeoutspeed = 11;	// 0.8 seconds
		holdlength = 150;	// 5 seconds
		buffer = new byte[(gridy*gridx)*2 + 3];		// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 							// start byte
		buffer[1] = (byte)0;							// command byte
		buffer[buffer.length-1] = (byte)254; 			// end byte
		soundID = -1;
		
		lights = new Light[6*28];							// empty array
		int count = 0;										// count x and y
		for(int y=0; y<28; y++){							// for each y position
			for(int x = 0; x<6; x++){						// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				lights[count].setValue(0);
				count++;
			}
		}
	}

	public void start(){
		holdtimer = 0;
		fadein = true;
		hold = false;
		fadeout = false;
		// launch sound
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,false,1.0f,10000,"quickcolor");
	}
	
	public void stop(){
		// stop sound
		ConnectionMain.soundController.killSound(soundID);
	}
	
	public byte[] draw(){
		if(hold){
			if(holdtimer < holdlength){
				holdtimer++;
			} else {
				holdtimer = 0;
				hold = false;
				fadeout = true;
			}
		} else {
			if(fadein){
				if(blue < targetblue){
					blue += fadeinspeed;
				} else {
					blue = targetblue;
				}
				if(red < targetred){
					red += fadeinspeed;
				} else {
					red = targetred;
				}
				if(blue == targetblue && red == targetred){
					hold = true;
					fadein = false;
				}
			} else if(fadeout){
				if(blue > 0){
					blue -= fadeoutspeed;
				} else {
					blue = 0;
				}
				if(red > 0){
					red -= fadeoutspeed;
				} else {
					red = 0;
				}
			}
		}
		for(int i=0; i<lights.length; i++){							// process each light
			lights[i].setValue(red, blue);
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];									// red
			buffer[i*2 + 3] = pair[1];									// blue
		}
		return buffer;
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
}
