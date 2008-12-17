package net.electroland.connection.animation;

import java.util.Iterator;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class TrackingDots {
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, element, row;					// grid position stuff
	public int xpos, ypos;
	public int soundx, soundy;
	public int[] soundloc;
	public boolean indicatorOn = false;
	public int soundID;
	public String soundFile;
	public int gain = 1;
	
	public TrackingDots(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		soundFile = ConnectionMain.properties.get("soundEnterCell");
	}
	
	public byte[] draw(){
		Iterator <Person> itr = ConnectionMain.personTracker.getPersonIterator();
		int compensation = Integer.parseInt(ConnectionMain.properties.get("forwardCompensation"));
		while (itr.hasNext()){
			Person person = itr.next();
			//int[] loc = peoplelist[i].getIntLoc();
			int[] loc = person.getForwardIntLoc(compensation);
			xpos = loc[0];
			ypos = loc[1];
			element = xpos + ypos*gridx;
			if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
				lights[element].setValue(253,0);			// make light active
				if(person.element != element){		// new position
					if(indicatorOn){
						if(person.yvec > 0 && element + gridx < lights.length){ // moving up
							lights[element + gridx].setValue(255, 255);					// forward indicator
						} else if (person.yvec < 0 && element - gridx > 0) {	// moving down
							lights[element - gridx].setValue(255, 255);					// forward indicator
						}
					}
				
					// send sound when light is switched RIGHT HERE
//					try{
//						soundID = ConnectionMain.soundController.newSoundID();
//						soundloc = person.getNearestSpeaker();
//						ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundloc[0]+" "+soundloc[1]+" 0 "+gain);
//					} catch(NullPointerException e){
//						System.err.println("sound output error: "+e);
//					}
					person.element = element;
				}
			}
			//person.checkNeighbors();
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		
		return buffer;
	}
	
}
