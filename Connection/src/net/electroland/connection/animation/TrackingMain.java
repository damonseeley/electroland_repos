package net.electroland.connection.animation;

import java.util.Collection;
import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class TrackingMain implements Animation{
	
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, element, row;					// grid position stuff
	public int soundx, soundy;								// speaker location
	private int throb;										// throbbing element brightness
	private boolean throbdown;							// direction of throbbing fade;
	//private int[] rows, newrows;							// rows occupied by people
	private int[] rowtuple;
	private int[][] rowdata, newrowdata;					// rows occupied by people and the direction headed
	private RowSparkle[] waves;							// collision effects
	private boolean colliding;
	private int indicatorMode = 0;
	public int soundID;
	public String soundFile;
	public String soundFileA;
	public String soundFileB;
	public String soundFileC;
	public String soundFileD;
	public String[] soundFileList;
	public String soundFileRow;
	public int soundrowID;
	public int[] soundloc;
	public int gain = 1;
	public ConcurrentHashMap<Integer, Blast> blasts = new ConcurrentHashMap<Integer, Blast>();
	public ConcurrentHashMap<Integer, Particle> particles = new ConcurrentHashMap<Integer, Particle>();
	public ConcurrentHashMap<Integer, Explosion> explosions = new ConcurrentHashMap<Integer, Explosion>();
	public Blast[] blastlist;
	public Particle[] particlelist;
	public Explosion[] explosionlist;
	public Pong pong;
	public boolean pongmode = false;
	public int blastcounter = 0;
	public int particlecounter = 0;
	public int explosioncounter = 0;
	public float tempvec;
	private int duration;
	
	public TrackingMain(Light[] _lights, int duration){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		this.duration = duration;
		waves = new RowSparkle[28];							// collision effect for each row
		throb = 255;
		throbdown = true;
		for(int i=0; i<waves.length; i++){
			//waves[i] = new RowWave(i*gridx);
			//waves[i] = new RowWave(i);
			waves[i] = new RowSparkle(i);
		}
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		//soundFile = "tracking_bars.wav";
		//soundFile = "subtle_tech_interface_11.wav";
		//soundFileA = ConnectionMain.properties.get("soundCollisionA");
		//soundFileB = ConnectionMain.properties.get("soundCollisionB");
		//soundFileC = ConnectionMain.properties.get("soundCollisionC");
		//soundFileD = ConnectionMain.properties.get("soundCollisionD");
		soundFileList = new String[8];
		soundFileList[0] = ConnectionMain.properties.get("soundCollisionA");
		soundFileList[1] = ConnectionMain.properties.get("soundCollisionB");
		soundFileList[2] = ConnectionMain.properties.get("soundCollisionC");
		soundFileList[3] = ConnectionMain.properties.get("soundCollisionD");
		soundFileList[4] = ConnectionMain.properties.get("soundCollisionE");
		soundFileList[5] = ConnectionMain.properties.get("soundCollisionF");
		soundFileList[6] = ConnectionMain.properties.get("soundCollisionG");
		soundFileList[7] = ConnectionMain.properties.get("soundCollisionH");
		soundFile = ConnectionMain.properties.get("soundTrackingGlobal");
		soundFileRow = ConnectionMain.properties.get("soundEnterCell");
		pong = new Pong(lights);
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
	public byte[] draw(){
		if(pongmode){
			// TEST ONLY
			//return pong.draw(peoplelist);
		}
		
		if(throb > 200 && throbdown){
			throb -= 5;
		} else if(throb <= 200 && throbdown){
			throbdown = false;
		} else if(throb < 255 && !throbdown){
			throb += 5;
		} else if(throb >= 255 && !throbdown){
			throbdown = true;
		}
		//rows = new int[0];
		rowdata = new int[0][2];
		rowtuple = new int[2];
		colliding = false;
		// THIS DOES ALL THE INDICATOR ADJUSTMENTS BASED ON PERSON COUNT
		/*
		if(peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdA"))){
			indicatorMode = 1;
			drawBackground(peoplelist);	// ambient background to fill negative space
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdA")) && peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdB"))){
			indicatorMode = 2;
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdB")) && peoplelist.length < Integer.parseInt(ConnectionMain.properties.get("personThresholdC"))){
			indicatorMode = 3;
		} else if(peoplelist.length >= Integer.parseInt(ConnectionMain.properties.get("personThresholdC"))){
			
		}
		*/
		Iterator<Person> p = ConnectionMain.personTracker.getPersonIterator();
		float compensation = Float.parseFloat(ConnectionMain.properties.get("forwardCompensation"));
		while (p.hasNext()){
			Person person = p.next();
			int[] loc = person.getForwardIntLoc(compensation);	// check location
			element = loc[1]*gridx + loc[0];
			//row = loc[1]*gridx;
			//row = loc[1];
			rowtuple[0] = loc[1];					// y value of row
			rowtuple[1] = person.getVec();	// vector of person in row (0 for negative, 1 for positive)
			if(element < lights.length && element >= 0){	// when people go out of range they need to be omitted
				if(loc[1] == 0 && person.yvec > 0 || loc[1] == 27 && person.yvec < 0){
					// BLAST OFF
					if(!person.inside){
						blasts.put(new Integer(blastcounter), new Blast(lights, blasts, blastcounter, person.id, loc[0], person.y));
						blastcounter++;
					}
				}
				if(person.element != element){		// person enters a new cell
					// SEND NEW CELL SOUND HERE
				}
				if(person.lastlighty != person.lighty){
					//System.out.println("ROWCHANGE, should hear sound");
					// SEND NEW ROW SOUND HERE
					//soundrowID = ConnectionMain.soundController.newSoundID();
					//soundloc = person.getNearestSpeaker();
					//ConnectionMain.soundController.send("simple instance"+soundrowID+" "+soundFileRow+" "+soundloc[0]+" "+soundloc[1]+" 0 "+0.8);
					//ConnectionMain.soundController.send("simple instance"+soundrowID+" "+soundFileRow+" "+soundloc[0]+" "+soundloc[1]+" 0 "+gain);
					person.lastlighty = person.lighty;
				}
				//for(int r=0; r<rows.length; r++){
				//	if(rows[r] == row){						// rows collide
				//		colliding = true;
				//	}
				//}
				for(int r=0; r<rowdata.length; r++){
					if(rowdata[r][0] == rowtuple[0] && rowdata[r][1] != rowtuple[1]){	// if in the same row and going different directions
						colliding = true;
					}
				}
				//newrows = new int[rows.length+1];
				//System.arraycopy(rows, 0, newrows, 0, rows.length);
				//rows = newrows;
				//rows[rows.length-1] = row;
				
				newrowdata = new int[rowdata.length+1][2];
				System.arraycopy(rowdata, 0, newrowdata, 0, rowdata.length);
				rowdata = newrowdata;
				rowdata[rowdata.length-1][0] = rowtuple[0];
				rowdata[rowdata.length-1][1] = rowtuple[1];

				if(colliding){
					//System.out.println(row/gridx);
					//waves[row/gridx].active = true;
					//waves[row].active = true;
					if(!waves[rowtuple[0]].active){
						waves[rowtuple[0]].activate();
					}
					//lights[row+n].setValue((int)(Math.random()*253),(int)(Math.random()*253));	// random test					
				} else {
					if(indicatorMode == 0){
						// SINGLE BAR MODE
						for(int n=0; n<gridx; n++){									// draw row above person
							lights[(rowtuple[0]*gridx)+n].setBlue(253);
						}
					} else	if(indicatorMode == 1){
						// DOUBLE BAR MODE
						for(int n=0; n<gridx; n++){									// draw row above person
							lights[(rowtuple[0]*gridx)+n].setValue(0, 253);
						}
						if(person.yvec > 0 && element + gridx < lights.length){ // moving up
							for(int n=0; n<gridx; n++){								// draw row
								lights[((rowtuple[0]+1)*gridx)+n].setBlue(253);			// blue 6 light line
							}
						} else if (person.yvec < 0 && element - gridx > 0) {	// moving down
							for(int n=0; n<gridx; n++){								// draw row
								lights[((rowtuple[0]-1)*gridx)+n].setBlue(253);			// blue 6 light line
							}
						}
					} else if(indicatorMode == 2){										// illuminates light in front of person
						// BAR + FRONT AND BACK LIGHTS
						for(int n=0; n<gridx; n++){									// draw row above person
							lights[(rowtuple[0]*gridx)+n].setBlue(253);
						}
						if(person.yvec > 0 && element + gridx < lights.length){ // moving up
							lights[element + gridx].setValue(0, 255);					// forward indicator
						} else if (person.yvec < 0 && element - gridx > 0) {	// moving down
							lights[element - gridx].setValue(0, 255);					// forward indicator
						}
						if(person.yvec > 0 && element - gridx >= 0){				// moving up
							lights[element - gridx].setValue(0, 255);					// behind indicator
						} else if(person.yvec < 0 && element + gridx < lights.length){	// moving down
							lights[element + gridx].setValue(0, 255);					// behind indicator
						}
					} else if(indicatorMode == 3){
						// CROSS MODE
						if(person.yvec > 0 && element + gridx < lights.length){ // moving up
							lights[element + gridx].setValue(0, 255);					// forward indicator
						} else if (person.yvec < 0 && element - gridx > 0) {	// moving down
							lights[element - gridx].setValue(0, 255);					// forward indicator
						}
						if(person.yvec > 0 && element - gridx > 0){				// moving up
							lights[element - gridx].setValue(0, 255);					// behind indicator
						} else if(person.yvec < 0 && element + gridx < lights.length){	// moving down
							lights[element + gridx].setValue(0, 255);					// behind indicator
						}
						if(loc[0] > 0 && element-1 >= 0){							// left light
							lights[element-1].setValue(0, 255);
						}
						if(loc[0] < 5 && element+1 < lights.length){				// right light
							lights[element+1].setValue(0, 255);
						}
					}
					lights[element].setBlue(throb);			// DIRECTLY ABOVE PERSON
				}
				person.inside = true;
			} else {
				person.inside = false;
			}
			colliding = false;
		}
		
		for(int i=0; i<waves.length; i++){					// row effects
			if(waves[i].active){							// if active row...
				//waves[i].draw(lights);						// draw it
				waves[i].draw();
			}
		}
		
		blastlist = new Blast[blasts.size()];				// create empty array
		blasts.values().toArray(blastlist);					// populate array with blast objects
		for(int i=0; i<blastlist.length; i++){				// for each active blast...
			blastlist[i].move();
			if(i != blastlist.length-1){					// if not the last (or only) blast...
				for(int n=i+1; n<blastlist.length; n++){	// check every other active blast...
					if(blastlist[i].gy == blastlist[n].gy && blastlist[i].gx == blastlist[n].gx){	// if two blasts share the same light...
						System.out.println("BLAST COLLISION");
						blasts.remove(new Integer(blastlist[i].id));
						blasts.remove(new Integer(blastlist[n].id));
						explosions.put(new Integer(explosioncounter), new Explosion(lights, explosioncounter, blastlist[i].gx, blastlist[i].gy));
						explosioncounter++;
					}
				}
			}
			p = ConnectionMain.personTracker.getPersonIterator();
			while (p.hasNext()){
				Person person = p.next();
				int[] loc = person.getForwardIntLoc(compensation);	// check location
				//System.out.println(blastlist[i].gy +" "+ loc[1]);
				if(indicatorMode < 3){
					if(blastlist[i].gy == loc[1] && blastlist[i].personID != person.id){		// if blast hit row and not launched by this person...
						explosions.put(new Integer(explosioncounter), new Explosion(lights, explosioncounter, blastlist[i].gx, blastlist[i].gy));
						explosioncounter++;
						blasts.remove(new Integer(blastlist[i].id));
					}
				} else {
					if(blastlist[i].gy == loc[1] && blastlist[i].gx == loc[0] && blastlist[i].personID != person.id){
						explosions.put(new Integer(explosioncounter), new Explosion(lights, explosioncounter, blastlist[i].gx, blastlist[i].gy));
						explosioncounter++;
						blasts.remove(new Integer(blastlist[i].id));
					}
				}
			}
			//blastlist[i].move(); //moved to the front
		}
		blastlist = new Blast[blasts.size()];				// EMPTY ARRAY POR FAVOR!

		explosionlist = new Explosion[explosions.size()];
		explosions.values().toArray(explosionlist);
		for(int i=0; i<explosionlist.length; i++){
			if(!explosionlist[i].done){
				explosionlist[i].expand();
			} else {
				explosions.remove(new Integer(explosionlist[i].id));
			}
		}
		
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	public void drawBackground(Collection <Person> peopleCollection){
		// step 1: fill entire grid with light values
		int color = 255;
		for(int i=0; i<lights.length; i++){
			color = (int)(Math.random()*150);		// low purple value
			lights[i].setValue(color, color);
			//System.out.println(i);
		}
		// step 2: ramp down light values near people rows
		Iterator <Person> itr = peopleCollection.iterator();
		while (itr.hasNext()){
			Person person = itr.next();
			try{
				int loc[] = person.getIntLoc();
				if(loc[1]-1 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]-1)*gridx + n < lights.length && (loc[1]-1)*gridx + n >= 0){
							lights[(loc[1]-1)*gridx + n].setValue(lights[(loc[1]-1)*gridx + n].red/3, lights[(loc[1]-1)*gridx + n].blue/3);
						}
					}
				}
				if(loc[1]-2 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]-2)*gridx + n < lights.length && (loc[1]-2)*gridx + n >= 0){
							lights[(loc[1]-2)*gridx + n].setValue(lights[(loc[1]-2)*gridx + n].red/2, lights[(loc[1]-2)*gridx + n].blue/2);
						}
					}
				}
				if(loc[1]-3 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]-3)*gridx + n < lights.length && (loc[1]-3)*gridx + n >= 0){
							lights[(loc[1]-3)*gridx + n].setValue((int)(lights[(loc[1]-3)*gridx + n].red*0.75), (int)(lights[(loc[1]-3)*gridx + n].blue*0.75));
						}
					}
				}
				
				if(loc[1]+1 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]+1)*gridx + n < lights.length && (loc[1]+1)*gridx + n >= 0){
							lights[(loc[1]+1)*gridx + n].setValue(lights[(loc[1]+1)*gridx + n].red/3, lights[(loc[1]+1)*gridx + n].blue/3);
						}
					}
				}
				if(loc[1]+2 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]+2)*gridx + n < lights.length && (loc[1]+2)*gridx + n >= 0){
							lights[(loc[1]+2)*gridx + n].setValue(lights[(loc[1]+2)*gridx + n].red/2, lights[(loc[1]+2)*gridx + n].blue/2);
						}
					}
				}
				if(loc[1]+3 >= 0){
					for(int n=0; n<gridx; n++){
						if((loc[1]+3)*gridx + n < lights.length && (loc[1]+3)*gridx + n >= 0){
							lights[(loc[1]+3)*gridx + n].setValue((int)(lights[(loc[1]+3)*gridx + n].red*0.75), (int)(lights[(loc[1]+3)*gridx + n].blue*0.75));
						}
					}
				}
			}catch(Exception e){
				e.printStackTrace();
			}
		}
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Tracking Main");
		/*
		// no sound file selected in properties, so let's not bother max.
		ConnectionMain.properties.put("blueFadeLength", "0.8");
		try {
			soundID = ConnectionMain.soundController.newSoundID();
			ConnectionMain.soundController.globalSound(soundID,soundFile,true,1.0f,10000);
		} catch (NullPointerException e) {
			e.printStackTrace();
		}
		*/
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Tracking Main " + soundID);
		ConnectionMain.soundController.killSound(soundID);
	}
	
	public void reset(){
		blasts = new ConcurrentHashMap<Integer, Blast>();
		particles = new ConcurrentHashMap<Integer, Particle>();
		explosions = new ConcurrentHashMap<Integer, Explosion>();
		blastlist = new Blast[blasts.size()];
		particlelist = new Particle[particles.size()];
		explosionlist = new Explosion[explosions.size()];
		blastcounter = 0;
		particlecounter = 0;
		explosioncounter = 0;
		rowdata = new int[0][2];
		colliding = false;
		for(int i=0; i<waves.length; i++){
			waves[i].active = false;
		}
	}
	
	
	
	
	
	
	public class RowSparkle{
		public int row;						// location of wave effect
		public int age;
		public int rowmembers;					// number of people in row
		public boolean active;
		public int[] values = new int[6];		// values of lights in row
		public int[] newvalues = new int[6];
		public boolean playing = false;
		public int soundID;
		public String soundFile;
		public int framecounter;
		
		public RowSparkle(int _row){
			row = _row;
			age = 0;
			framecounter = 0;
			active = false;
			/*
			for(int i=0; i<values.length; i++){
				values[i] = (int)(Math.random()*253);
			}
			*/
			values[0] = 0;
			for(int i=1; i<values.length-1; i++){
				values[i] = 253;
			}
		}
		
		public void activate(){
			active = true;
			// randomize sound file to be played on collision
			/*
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
			*/
			soundFile = ConnectionMain.soundController.randomSound(soundFileList);
			// send sound TO BOTH SPEAKERS IN ROW when collision is first triggered
			try{
				// WHY DOES IT NOT WORK IN ONE METHOD?!
				soundy = (int)Math.floor(row/2);
				if(soundy < 1){
					soundy = 1;
				} else if(soundy > 12){
					soundy = 12;
				}
				//soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
				//ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundy+" 1 0 "+gain);
				//soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
				//ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundy+" 2 0 "+gain);
				
				
				// send sound TO BOTH SPEAKERS IN ROW when collision is first triggered
				ConnectionMain.soundController.playSimpleSound(soundFile, 0, row, 0.8f, "rowCollisionTop");
				ConnectionMain.soundController.playSimpleSound(soundFile, 5, row, 0.8f, "rowCollisionBottom");
				
			} catch(NullPointerException e){
				System.err.println("sound output error: "+e);
			}
		}
		
		public void draw(){
			if(framecounter == 1){
				newvalues = new int[6];
				System.arraycopy(values, 0, newvalues, 1, values.length-1);
				newvalues[0] = values[values.length-1];
				values = newvalues;
				//values[0] = (int)(Math.random()*253);
				framecounter = 0;
			} else {
				framecounter++;
			}
			
			for(int n=0; n<values.length; n++){
				lights[(row*gridx)+n].setValue(values[n],0);
			}
			
			if(age > 5){
				active = false;
				rowmembers = 0;
				for(int i=0; i<rowdata.length; i++){
					if(rowdata[i][0] == row){
						rowmembers++;
					}
				}
				if(rowmembers > 1){
					active = true;
				}
				age = 0;
			} else {
				age += 1;
			}
		}		
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
		public int soundID;
		public String soundFile;
		
		public RowWave(int _row){
			row = _row;
			age = 0;
			active = false;
		}
		
		public void draw(Light[] lights){
			theta += 0.4;
			brightness = theta;
			for(int n=0; n<gridx; n++){
				//lights[row+n].setValue((int)(128 + Math.sin(brightness)*amplitude),0); // top to bottom wave
				lights[(row*gridx)+n].setValue((int)(128 + Math.sin(brightness)*amplitude),0); // top to bottom wave
				//lights[row+n].setValue((int)(128 + Math.sin(brightness)*amplitude),(int)(Math.random()*amplitude));
				brightness += spacing;
			}
			if(age == 0){
				soundFile = ConnectionMain.soundController.randomSound(soundFileList);
				// send sound TO BOTH SPEAKERS IN ROW when collision is first triggered
				soundy = (int)Math.floor(row/2);
				if(soundy < 1){
					soundy = 1;
				} else if(soundy > 12){
					soundy = 12;
				}
				ConnectionMain.soundController.playSimpleSound(soundFile, 1, soundy, gain, "rowCollisionTop");
				ConnectionMain.soundController.playSimpleSound(soundFile, 2, soundy, gain, "rowCollisionBottom");
				
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

