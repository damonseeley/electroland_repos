package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;

public class Wave implements Animation {
	public int gridx, gridy;
	public int frameCount, element;
	float theta = 0;								// speed
	float amplitude = 128;							// -128 to 128
	float period = 100;								// spacing in lights
	float spacing = (float)(Math.PI*2 / period);
	float brightness;
	public byte[] buffer;
	public int soundID;
	public String soundFile;
	public int gain = 1;
	private int duration;
	
	public Wave(int _gridx, int _gridy, int duration, String soundFile){
		this.duration = duration;
		this.soundFile = soundFile;
		gridx = _gridx;
		gridy = _gridy;
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		//soundFile = ConnectionMain.properties.get("soundWaveGlobal");
		soundID = -1;
	}
	public int getDefaultDuration(){
		return duration;
	}
	public byte[] draw(){
		theta += 0.2;
		brightness = theta;
		for(int y=0; y<gridy; y++){
			for(int x=0; x<gridx; x++){
				element = x+y*gridx;
				// THIS DOES THE COOL BLUE BAR BECAUSE OF AN OFFSET ERROR SENDING A BYTE OF 254 or 255
				buffer[element*2 + 2] = (byte)(128 + Math.sin(brightness)*amplitude);	// red
				buffer[element*2 + 3] = (byte)(((float)x/gridx)*255);					// blue
			}
			brightness += spacing;
		}
		return buffer;
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Wave");
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,true,1,10000,"wave");
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Wave " + soundID);
		ConnectionMain.soundController.killSound(soundID);
	}
}
