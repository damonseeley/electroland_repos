package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class RandomFill implements Animation{

	public int gridx, gridy;							// grid dimensions
	public byte[] buffer, pair;						// final output
	public Light[] lights;								// light objects
	public int[] lightlist, activelightlist;
	public int soundID;
	public String soundFile;
	private int duration;
	
	public RandomFill(int duration){
		this.duration = duration;
		gridx = 6;
		gridy = 28;
		buffer = new byte[(gridy*gridx)*2 + 3];		// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 							// start byte
		buffer[1] = (byte)0;							// command byte
		buffer[buffer.length-1] = (byte)254; 			// end byte
		soundFile = ConnectionMain.properties.get("soundRandomFill");
		soundID = -1;
		
		lights = new Light[6*28];							// empty array
		lightlist = new int[6*28];
		activelightlist = new int[0];
		int count = 0;										// count x and y
		for(int y=0; y<28; y++){							// for each y position
			for(int x = 0; x<6; x++){						// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				lights[count].setValue(0);
				lightlist[count] = count;					// holds all light positions
				count++;
			}
		}
	}
	
	public void start(){
		lights = new Light[6*28];							// empty array
		lightlist = new int[6*28];
		activelightlist = new int[0];
		int count = 0;										// count x and y
		for(int y=0; y<28; y++){							// for each y position
			for(int x = 0; x<6; x++){						// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				lights[count].setValue(0);
				lightlist[count] = count;					// holds all light positions
				count++;
			}
		}
		// launch sound
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,false,1.0f,10000,"randomfill");
	}
	
	public void stop(){
		// stop sound
		//ConnectionMain.soundController.killSound(soundID);
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
	private void getRandomLight(){
		int[] newactivelightlist = new int[activelightlist.length+1];
		System.arraycopy(activelightlist, 0, newactivelightlist, 0, activelightlist.length);	// copy over currently active lights
		int num = (int)(Math.random()*lightlist.length);								// get random position in light list
		newactivelightlist[newactivelightlist.length-1] = lightlist[num];				// append number of light object
		activelightlist = newactivelightlist;
		
		int[] newlightlist = new int[lightlist.length-1];								// create empty light list
		System.arraycopy(lightlist, 0, newlightlist, 0, num);							// copy everything before position
		System.arraycopy(lightlist, num+1, newlightlist, num, lightlist.length-(num+1));// copy everything after position
		lightlist = newlightlist;
	}
	
	public byte[] draw(){
		if(lightlist.length > 1){
			// keep getting new lights, 2 at a time
			getRandomLight();
			getRandomLight();
		}
		for(int i=0; i<activelightlist.length; i++){
			Light light = lights[activelightlist[i]];
			// WHY DOES THIS FLICKER?!
			if(light.red < 253){
				light.setValue(light.red + 7);
			} else {
				light.setValue(253);
			}
		}
		for(int i=0; i<lights.length; i++){							// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];									// red
			buffer[i*2 + 3] = pair[1];									// blue
		}
		return buffer;
	}
}
