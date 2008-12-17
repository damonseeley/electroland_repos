package net.electroland.connection.animation;

import java.util.concurrent.ConcurrentHashMap;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Clash {

	public Light[] lights;
	public byte[] buffer;
	public byte[] pair;
	public Wave waveA, waveB;
	public Splash splash;
	public ConcurrentHashMap<Integer, Wave> waves = new ConcurrentHashMap<Integer, Wave>();
	public ConcurrentHashMap<Integer, Splash> splashes = new ConcurrentHashMap<Integer, Splash>();
	public Wave[] wavelist;
	public Splash[] splashlist;
	public int wavecounter;
	public int splashcounter;
	public int mandatorywait = 2;
	public int waitcounter = 0;
	public int soundID;
	public String soundFile;
	
	public Clash(Light[] _lights){
		lights = _lights;									// used to automate fading process
		//waveA = new Wave(-0.01f, "red");
		//waveB = new Wave(0.01f, "blue");
		//splash = new Splash();
		buffer = new byte[(28*6)*2 + 3];					// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		soundFile = ConnectionMain.properties.get("soundClashGlobal");
		soundID = -1;
	}
	
	public byte[] draw(){
		if(waitcounter == mandatorywait){
			if(Math.random() > 0.9){							// 1 in 10 chance of creating a new wave pair
				waves.put(new Integer(wavecounter), new Wave(wavecounter, wavecounter+1, -(float)(Math.random()*0.02)-0.005f, "red"));
				wavecounter++;
				waves.put(new Integer(wavecounter), new Wave(wavecounter, wavecounter-1, (float)(Math.random()*0.02)+0.005f, "blue"));
				wavecounter++;
			} else if(Math.random() < 0.1){					// 1 in 10 chance of creating a new wave pair (swapped colors)
				waves.put(new Integer(wavecounter), new Wave(wavecounter, wavecounter+1, -(float)(Math.random()*0.02)-0.005f, "blue"));
				wavecounter++;
				waves.put(new Integer(wavecounter), new Wave(wavecounter, wavecounter-1, (float)(Math.random()*0.02)+0.005f, "red"));
				wavecounter++;
			}
			waitcounter = 0;
		} else {
			waitcounter++;
		}
		
		wavelist = new Wave[waves.size()];
		waves.values().toArray(wavelist);
		for(int i=0; i<wavelist.length; i++){
			wavelist[i].move();
		}
		
		splashlist = new Splash[splashes.size()];
		splashes.values().toArray(splashlist);
		for(int i=0; i<splashlist.length; i++){
			splashlist[i].draw();
		}
		
		
		/*
		if(Math.abs(waveA.y - waveB.y) > 0.01){
			// animate each wave until they smash against each other
			waveA.move(lights);
			waveB.move(lights);
		} else {
			// when they smash make a Splash
			if(splash.done){
				reset();
			} else {
				splash.draw(lights);
			}
		}
		*/
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		
		return buffer;
	}
	
	public void reset(){
		//waveA = new Wave(-0.01f, "red");
		//waveB = new Wave(0.01f, "blue");
		//splash = new Splash();
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Clash");
		try {
			soundID = ConnectionMain.soundController.newSoundID();
			ConnectionMain.soundController.globalSound(soundID,soundFile,true,1.0f,10000,"clash");
		} catch (NullPointerException e) {
			e.printStackTrace();
		}
		
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Clash " + soundID);
		ConnectionMain.soundController.killSound(soundID);
	}
	
	
	
	
	public class Wave{
		public int id, partner;
		public float y;
		public float yvec;
		public int gy, element;
		public String color;
		
		public Wave(int _id, int _partner, float _yvec, String _color){
			id = _id;
			partner = _partner;
			yvec = _yvec;		// speed
			if(yvec < 0){
				y = 1;
			} else {
				y = 0;
			}
			color = _color;		// text of color
		}
		
		public void move(){
			// wave needs to move up or down across entire width of light grid.
			// location is a normalized position in grid.
			
			if(waves.containsKey(id) && waves.containsKey(partner)){
				if(Math.abs(y - waves.get(partner).y) < 0.01){	// if waves collide
					waves.remove(new Integer(id));				// destroy waves 
					waves.remove(new Integer(partner));
					// create a splash at this location
					splashes.put(new Integer(splashcounter), new Splash(splashcounter, y));
					splashcounter++;
				}
			}
			
			y += yvec;										// moves up or down
			gy = Math.round(y*28);							// position relative to light grid long-ways axis
			if(gy == 28){
				gy = 27;
			}
			element = gy * 6;								// ypos * grid width
			for(int i=element; i<element+6; i++){			// for each light in row
				if(element < lights.length && element >= 0){
					if(color == "red"){
						lights[i].setRed(255);
					} else if (color =="blue"){
						lights[i].setBlue(255);
					}
				}
			}
		}
		
	}
	
	
	
	
	public class Splash{
		public boolean done;
		public Particle[] particles;
		private int spentparticles;
		public int soundID;
		public String soundFile;
		public boolean playing;
		public int gain = 1;
		public float y;
		public int id;
		
		public Splash(int _id, float _y){
			id = _id;
			y = _y;
			done = false;
			particles = new Particle[12];
			spentparticles = 0;
			for(int i=0; i<6; i++){		// particles going up
				particles[i] = new Particle(i, y, (float)(-0.05*Math.random() - 0.01));
			}
			for(int i=6; i<12; i++){		// particles going down
				particles[i] = new Particle(i-6, y, (float)(0.05*Math.random() + 0.01));
			}
			soundFile = ConnectionMain.properties.get("soundClash");
			soundID = -1;
			playing = false;
		}
		
		public void draw(){
			if(!playing){
				playing = true;
				//soundID = ConnectionMain.soundController.newSoundID();
				//ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" 6 1 0 "+gain);
				//soundID = ConnectionMain.soundController.newSoundID();
				//ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" 6 2 0 "+gain);
				
			}
			// scatter lights from center of grid, rows 14 x 15
			for(int i=0; i<particles.length; i++){
				if(!particles[i].done){
					particles[i].move(lights);
				} else {
					spentparticles++;
				}
			}
			if(spentparticles == 12){
				done = true;
				splashes.remove(id);
			}
		}
		
		
		/*
		public class Particle{
			float y, yvec;
			int x, element, gy;
			public boolean done;
			int age;
			
			public Particle(int _x, float _y, float _yvec){
				x = _x;
				y = _y;
				yvec = _yvec;
				age = 255;
			}
			
			public void move(Light[] lights){
				gy = Math.round(y*28);	// position relative to light grid long-ways axis
				if(gy == 28){
					gy = 27;
				} else if(gy < 0){
					gy = 0;
				}
				element = gy*6 + x;
				if(element >= 0 && element < 28*6){
					//lights[element].setValue((int)(Math.random()*age));	// flickering mode
					lights[element].setValue(age);							// smooth fade mode
				}
				
				age -= 5;
				if(age <= 0){
					done = true;
					spentparticles += 1;
				} else {
					yvec *= 0.9;				
					if(yvec < 0 && y > 0){
						y += yvec;
					} else if(yvec > 0 && y < 1){
						y += yvec;
					} else {
						done = true;
						spentparticles += 1;
					}
				}
			}
		}
		*/
	}
	
	
}