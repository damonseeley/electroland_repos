package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Matrix implements Animation{

	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy;								// grid position stuff
	public Chunk[] chunks;
	
	public int soundID;
	public String soundFile;
	private int duration;
	
	public Matrix(Light[] _lights, int duration, String soundFile){
		this.duration = duration;
		this.soundFile = soundFile;
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		chunks = new Chunk[25];
		for(int i=0; i<chunks.length; i++){
			chunks[i] = new Chunk(lights);
		}
		soundID = -1;
		//soundFile = ConnectionMain.properties.get("soundMatrixGlobal");
		
		// TEMP!!!!
		//start();
	}
	
	
	public int getDefaultDuration(){
		return duration;
	}
	
	public void start() {
		// start the matrix sound
		System.out.println("START: Matrix");
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,true,1,10000,"matrix");

	}
	
	public void stop() {
		// stop the matrix sound
		System.out.println("STOP: Matrix " + soundID);
		//ConnectionMain.soundController.killSound(soundID);
	}
	
	public byte[] draw(){
		for(int i=0; i<chunks.length; i++){
			chunks[i].move();
		}
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
}
