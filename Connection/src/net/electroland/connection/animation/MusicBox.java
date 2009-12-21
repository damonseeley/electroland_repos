package net.electroland.connection.animation;

import java.util.Iterator;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class MusicBox implements Animation {
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public Player player;
	public int gridx, gridy, element, row;					// grid position stuff
	public int xpos, ypos;
	public int frameCount;
	public int framesPerRefresh;
	public int notesoundID;
	public int soundID;
	public String soundFileA;
	public String soundFileB;
	public String soundFileC;
	public String soundFileD;
	public String soundFileE;
	public String soundFileF;
	public String soundFile;
	public int gain;
	private int duration;

	public MusicBox(Light[] _lights, int duration){
		this.duration = duration;
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(gridx*gridy)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		frameCount = 0;
		framesPerRefresh = 2;								//  sets tempo for playback
		player = new Player("blue");
		soundFileA = ConnectionMain.properties.get("soundMusicBoxA");
		soundFileB = ConnectionMain.properties.get("soundMusicBoxB");
		soundFileC = ConnectionMain.properties.get("soundMusicBoxC");
		soundFileD = ConnectionMain.properties.get("soundMusicBoxD");
		soundFileE = ConnectionMain.properties.get("soundMusicBoxE");
		soundFileF = ConnectionMain.properties.get("soundMusicBoxF");
		soundFile = ConnectionMain.properties.get("soundMusicBoxGlobal");
		soundID = -1;
		gain = 1;
	}
	public int getDefaultDuration(){
		return duration;
	}	
	public byte[] draw(){
		
		/*
		if(frameCount >= framesPerRefresh){
			player.move();
			frameCount = 0;
		} else {
			frameCount++;
			//System.out.println(frameCount);
		}
		float compensation = Float.parseFloat(ConnectionMain.properties.get("forwardCompensation"));	// one and only access per draw
		
		Iterator <Person> itr = ConnectionMain.personTracker.getPersonIterator();
		while (itr.hasNext()){
			Person person = itr.next();
			int[] loc = person.getForwardIntLoc(compensation);
			xpos = loc[0];
			ypos = loc[1];
			element = xpos + ypos*gridx;
			if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
				lights[element].setRed(253);				// make light active
				if(player.gy == ypos){
					// play sound above person
					if(xpos == 0){
						ConnectionMain.soundController.playSimpleSound(soundFileA, xpos, ypos, 1, "musicboxA");
					} else if(xpos == 1){
						ConnectionMain.soundController.playSimpleSound(soundFileB, xpos, ypos, 1, "musicboxB");
					} else if(xpos == 2){
						ConnectionMain.soundController.playSimpleSound(soundFileC, xpos, ypos, 1, "musicboxC");
					} else if(xpos == 3){
						ConnectionMain.soundController.playSimpleSound(soundFileD, xpos, ypos, 1, "musicboxD");
					} else if(xpos == 4){
						ConnectionMain.soundController.playSimpleSound(soundFileE, xpos, ypos, 1, "musicboxE");
					} else if(xpos == 5){
						ConnectionMain.soundController.playSimpleSound(soundFileF, xpos, ypos, 1, "musicboxF");
					}
				}
			}
		}
		*/

		float compensation = Float.parseFloat(ConnectionMain.properties.get("forwardCompensation"));	// one and only access per draw
		if(frameCount >= framesPerRefresh){
			player.move();
			frameCount = 0;
			
			Iterator <Person> itr = ConnectionMain.personTracker.getPersonIterator();
			while (itr.hasNext()){
				Person person = itr.next();
				int[] loc = person.getForwardIntLoc(compensation);
				xpos = loc[0];
				ypos = loc[1];
				element = xpos + ypos*gridx;
				if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
					//lights[element].setRed(253);				// make light active
					if(player.gy == ypos){
						// play sound above person
						if(xpos == 0){
							ConnectionMain.soundController.playSimpleSound(soundFileA, xpos, ypos, 1, "musicboxA");
						} else if(xpos == 1){
							ConnectionMain.soundController.playSimpleSound(soundFileB, xpos, ypos, 1, "musicboxB");
						} else if(xpos == 2){
							ConnectionMain.soundController.playSimpleSound(soundFileC, xpos, ypos, 1, "musicboxC");
						} else if(xpos == 3){
							ConnectionMain.soundController.playSimpleSound(soundFileD, xpos, ypos, 1, "musicboxD");
						} else if(xpos == 4){
							ConnectionMain.soundController.playSimpleSound(soundFileE, xpos, ypos, 1, "musicboxE");
						} else if(xpos == 5){
							ConnectionMain.soundController.playSimpleSound(soundFileF, xpos, ypos, 1, "musicboxF");
						}
					}
				}
			}
		} else {
			frameCount++;
			//System.out.println(frameCount);
		}
		
		Iterator <Person> itr = ConnectionMain.personTracker.getPersonIterator();
		while (itr.hasNext()){
			Person person = itr.next();
			int[] loc = person.getForwardIntLoc(compensation);
			xpos = loc[0];
			ypos = loc[1];
			element = xpos + ypos*gridx;
			if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
				lights[element].setRed(253);				// make light active
			}
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	public void reset(){
		frameCount = 0;
		player.y = 0;
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Music Box");
		ConnectionMain.properties.put("blueFadeLength", "0.95");
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,true,1.0f,10000,"musicbox");

	}
	
	public void stop() {
		// stop the global sound
		ConnectionMain.properties.put("blueFadeLength", "0.8");
		System.out.println("STOP: Music Box " + soundID);
		//ConnectionMain.soundController.killSound(soundID);
	}
	
	
	
	
	public class Player{
		public float y;
		public float yvec;
		public int gy, element;
		public String color;
		public int soundID;
		public String soundFile;
		public boolean playing;
		
		public Player(String _color){
			color = _color;		// text of color
			//soundFile = ConnectionMain.properties.get("soundMusicBoxPlayer");
			playing = false;
		}
		
		public void move(){
			if(!playing){
				playing = true;
				// send complex looped sound command
				//soundID = ConnectionMain.soundController.newSoundID();
				//ConnectionMain.soundController.moveSound(soundID, 1.5f, y);
			}
			// location is a normalized position in grid.
			//y += yvec;										// moves up or down
			//gy = Math.round(y*28);							// position relative to light grid long-ways axis
			gy++;
			if(gy >= 28){
				gy = 0;
			}
			element = (gy * 6);								// ypos * grid width
			//System.out.println(gy +" "+ element);
			for(int i=element; i<element+6; i++){			// for each light in row
				if(color == "red"){
					lights[i].setRed(255);
				} else if (color =="blue"){
					lights[i].setBlue(255);
				}
			}
		}
		
	}
	
}
