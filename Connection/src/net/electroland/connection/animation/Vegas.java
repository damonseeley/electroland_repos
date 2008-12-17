package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;

public class Vegas implements Animation {

	public int gridx, gridy;	// grid data
	public int red, blue;		// multiplier for intensity
	public byte[] buffer;
	public boolean colorshift;
	public int speed;
	private int soundID;
	public String soundFile;
	public int gain = 1;
	public boolean playing = false;
	private int duration;
	
	public Vegas(int _gridx, int _gridy, boolean _colorshift, int duration){
		this.duration = duration;
		gridx = _gridx;
		gridy = _gridy;
		colorshift = _colorshift;							// animate red/blue values
		red = 253;
		blue = 0;
		speed = 2;											// 5, 15, 17, and 51
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		soundID = -1;
		soundFile = ConnectionMain.properties.get("soundVegasGlobal");
	}
	public int getDefaultDuration(){
		return duration;
	}
	public byte[] draw(){
		if(colorshift){
			for(int i=2; i<buffer.length-1; i+=2){
				buffer[i] = (byte)(Math.random()*red);
				buffer[i+1] = (byte)(Math.random()*blue);
			}
			if(red == 253 && blue < 253){
				blue += speed;
			} else if(blue == 253 && red > 0){
				red -= speed;
			}
			if(red > 253){
				red = 253;
			} else if(red < 0){
				red = 0;
			}
			if(blue > 253){
				blue = 253;
			} else if(blue < 0){
				blue = 0;
			}

		} else {
			for(int i=2; i<buffer.length-1; i++){
				buffer[i] = (byte)(Math.random()*253);
			}
		}
		return buffer;
	}
	
	public void start() {
		red = 253;
		blue = 0;
		// start the global sound
		soundID = ConnectionMain.soundController.newSoundID();
		System.out.println("START: Vegas " + soundID);
		ConnectionMain.soundController.globalSound(soundID,soundFile,true,1,10000,"vegas");
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Vegas " + soundID);
		ConnectionMain.soundController.killSound(soundID);
	}
	
}
