package net.electroland.connection.animation;

import net.electroland.connection.core.Light;

public class ScreenSaver implements Animation {

	public int gridx, gridy;							// grid dimensions
	public byte[] buffer, pair;						// final output
	public Light[] lights;								// light objects
	//public int soundID;
	//public String soundFile;
	public int[] lightlist, activelightlist;
	private int duration;
	private int lightCount;
	private int fadeInLength, holdLength, fadeOutLength;
	private int lightspacing, lightdelay;
	
	public ScreenSaver(int duration, int lightCount, int fadeInLength, int holdLength, int fadeOutLength){
		this.duration = duration;
		this.lightCount = lightCount;
		this.fadeInLength = fadeInLength;
		this.holdLength = holdLength;
		this.fadeOutLength = fadeOutLength;
		lightdelay = 0;
		lightspacing = 10;	// frames apart
		gridx = 6;
		gridy = 28;
		buffer = new byte[(gridy*gridx)*2 + 3];		// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 							// start byte
		buffer[1] = (byte)0;							// command byte
		buffer[buffer.length-1] = (byte)254; 			// end byte
		//soundFile = ConnectionMain.properties.get("soundBiggestGlobal");
		//soundID = -1;
		
		lights = new Light[6*28];							// empty array
		lightlist = new int[6*28];
		activelightlist = new int[0];
		int count = 0;										// count x and y
		for(int y=0; y<28; y++){							// for each y position
			for(int x = 0; x<6; x++){						// for each x position
				lights[count] = new Light(count, x, y);		// create a new light
				lightlist[count] = count;					// holds all light positions
				count++;
			}
		}
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
	public void setDefaultDuration(int millis){
		duration = millis;
	}
	
	private void getRandomLight(){
		int[] newactivelightlist = new int[activelightlist.length+1];
		System.arraycopy(activelightlist, 0, newactivelightlist, 0, activelightlist.length);	// copy over currently active lights
		int num = (int)(Math.random()*lightlist.length);								// get random position in light list
		newactivelightlist[newactivelightlist.length-1] = lightlist[num];				// append number of light object
		activelightlist = newactivelightlist;
		lights[lightlist[num]].startFade(fadeInLength, holdLength, fadeOutLength);		// START FADING LIGHT WHEN RETRIEVED
		//System.out.println("new light");
		
		int[] newlightlist = new int[lightlist.length-1];								// create empty light list
		System.arraycopy(lightlist, 0, newlightlist, 0, num);							// copy everything before position
		System.arraycopy(lightlist, num+1, newlightlist, num, lightlist.length-(num+1));// copy everything after position
		lightlist = newlightlist;
	}
	
	public byte[] draw(){
		if(activelightlist.length < lightCount){
			if(lightdelay < lightspacing){
				lightdelay++;
			} else {
				getRandomLight();							// get new lights while there's still room in the space
				lightdelay = 0;
			}
		}
		for(int i=0; i<activelightlist.length; i++){
			if(!lights[activelightlist[i]].fading){			// if light is done fading...
				// remove light from active light list
				// put back in light list for reuse
			}
		}
		for(int i=0; i<lights.length; i++){							// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];									// red
			buffer[i*2 + 3] = pair[1];									// blue
		}
		return buffer;
	}
	
	public void start(){
		activelightlist = new int[0];
		lightlist = new int[6*28];
		for(int i=0; i<lights.length; i++){
			lightlist[i] = i;
			lights[i].fading = false;
			lights[i].setValue(0,0);
		}
		System.out.println("START: ScreenSaver");
		// no sound for screensaver
		//soundID = ConnectionMain.soundController.newSoundID();
		//ConnectionMain.soundController.globalSound(soundID,soundFile,false,1.0f,10000);
	}
	
	public void stop(){
		System.out.println("STOP: ScreenSaver");
		//ConnectionMain.soundController.killSound(soundID);
	}
	
}
