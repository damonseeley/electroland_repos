package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class TrackingBars {
	
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, element, row;					// grid position stuff
	public int soundx;
	public float soundy;								// speaker location
	private int[] rows, newrows;							// rows occupied by people
	private RowWave[] waves;								// collision effects
	private boolean colliding;
	public int soundID;
	public String soundFile;
	public String soundFileA;
	public String soundFileB;
	public String soundFileC;
	public String soundFileD;
	public int[] soundloc;
	public int gain = 1;
	
	public TrackingBars(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		waves = new RowWave[28];							// collision effect for each row
		for(int i=0; i<waves.length; i++){
			waves[i] = new RowWave(i*gridx);
		}
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		//soundFile = "tracking_bars.wav";
		//soundFile = "subtle_tech_interface_11.wav";
		soundFileA = ConnectionMain.properties.get("soundCollisionA");
		soundFileB = ConnectionMain.properties.get("soundCollisionB");
		soundFileC = ConnectionMain.properties.get("soundCollisionC");
		soundFileD = ConnectionMain.properties.get("soundCollisionD");
	}
	
	public byte[] draw(Person[] peoplelist){
		rows = new int[0];
		colliding = false;
		if(peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdA"))){
			
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdA")) && peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdB"))){
			
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdB")) && peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdC"))){
			
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdC"))){
			
		}
		for(int i=0; i<peoplelist.length; i++){
			int loc[] = peoplelist[i].getIntLoc();
			element = loc[1]*gridx + loc[0];
			row = loc[1]*gridx;
			//element = (int) (Math.round(peoplelist[i].y*gridy)*gridx + Math.round(peoplelist[i].x*gridx));	// light above person
			//row = (int) Math.round(peoplelist[i].y*gridy)*gridx;
			if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
				//lights[element].setValue(253,0);			// make light active
				//soundx = (int)(peoplelist[i].y*12) + 1;
				//soundy = peoplelist[i].x*3;
				/*
				if (peoplelist[i].x > 0.5){
					soundy = 1;
				} else {
					soundy = 2;
				}
				*/
				if(peoplelist[i].element != element){	// new position
					// send sound when light is switched RIGHT HERE
					/*
					try{
						soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
						soundloc = peoplelist[i].getNearestSpeaker();
						//ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundx+" "+soundy+" "+gain);
						ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundloc[0]+" "+soundloc[1]+" 0 "+gain);
					} catch(NullPointerException e){
						System.err.println("sound output error: "+e);
					}
					*/
				}
				for(int r=0; r<rows.length; r++){
					if(rows[r] == row){						// rows collide
						colliding = true;
					}
				}
				newrows = new int[rows.length+1];
				System.arraycopy(rows, 0, newrows, 0, rows.length);
				rows = newrows;
				rows[rows.length-1] = row;
				if(colliding){
					//System.out.println(row/gridx);
					waves[row/gridx].active = true;
					//lights[row+n].setValue((int)(Math.random()*253),(int)(Math.random()*253));	// random test					
				} else {
					for(int n=0; n<gridx; n++){			// draw row
						lights[row+n].setBlue(253);		// blue 6 light line
					}
				}
			}
			colliding = false;
		}
		
		for(int i=0; i<waves.length; i++){					// row effects
			if(waves[i].active){							// if active row...
				waves[i].draw(lights);						// draw it
			}
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	
	
	
	public class RowWave{
		public int row;						// location of wave effect
		public int age;
		public boolean active;
		public float theta = 0;								// speed
		public float amplitude = 128;							// -128 to 128
		public float period = 12;								// spacing in lights
		public float spacing = (float)(Math.PI*2 / period);
		public float brightness;
		
		public RowWave(int _row){
			row = _row;
			age = 0;
			active = false;
		}
		
		public void draw(Light[] lights){
			theta += 0.4;
			brightness = theta;
			for(int n=0; n<gridx; n++){
				lights[row+n].setValue((int)(128 + Math.sin(brightness)*amplitude),0); // top to bottom wave
				//lights[row+n].setValue((int)(128 + Math.sin(brightness)*amplitude),(int)(Math.random()*amplitude));
				brightness += spacing;
			}
			if(age == 0){
				// randomize sound file to be played on collision
				float random = (float)Math.random();
				if(random < 0.25){
					soundFile = soundFileA;
				} else if(random >= 0.25 && random < 0.5){
					soundFile = soundFileB;
				} else if(random >= 0.5 && random < 0.75){
					soundFile = soundFileC;
				} else if(random >= 0.75){
					soundFile = soundFileD;
				}
				// send sound TO BOTH SPEAKERS IN ROW when collision is first triggered
//				try{
//					soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
//					ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+(int)Math.floor((row/gridx)/2)+" 1 0 "+gain);
//					soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
//					ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+(int)Math.floor((row/gridx)/2)+" 2 0 "+gain);
//				} catch(NullPointerException e){
//					System.err.println("sound output error: "+e);
//				}
			}
			if(age > 30){
				active = false;
				age = 0;
			} else {
				age += 1;
			}
		}
	}
	
}
