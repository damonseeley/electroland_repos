package net.electroland.connection.animation;

import java.util.Iterator;
import java.util.concurrent.ConcurrentHashMap;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import net.electroland.connection.core.Person;

public class TrackingConnections implements Animation {
	public byte[] buffer, pair;							// final output
	public Light[] lights;									// light objects
	public int gridx, gridy, row;							// grid position stuff
	public int xpos, ypos;
	public int indicatorMode = 0;
	public ConcurrentHashMap<Integer, Blast> blasts = new ConcurrentHashMap<Integer, Blast>();
	public ConcurrentHashMap<Integer, Explosion> explosions = new ConcurrentHashMap<Integer, Explosion>();
	public ConcurrentHashMap<String, ColumnPair> columnPairs = new ConcurrentHashMap<String, ColumnPair>();
	public ConcurrentHashMap<String, HorizontalPair> rowPairs = new ConcurrentHashMap<String, HorizontalPair>();
	public ConcurrentHashMap<String, DiagonalPair> diagonalPairs = new ConcurrentHashMap<String, DiagonalPair>();
	public Blast[] blastlist;
	public Explosion[] explosionlist;
	public float compensation = 5;
	public int blastcounter = 0;
	public int explosioncounter = 0;
	public int soundx, soundy;
	public int[] soundloc;
	public int soundID;
	public String soundFile;
	public String[] soundFileList;
	public int gain = 1;
	public boolean dashedMode;
	public int mindist = 2;
	private int duration;
	private int rowPairThreshold = Integer.valueOf(ConnectionMain.properties.get("rowPairThreshold"));
	private int diagPairThreshold = Integer.valueOf(ConnectionMain.properties.get("diagPairThreshold"));
	
	public TrackingConnections(Light[] _lights, int duration){
		this.duration = duration;
		gridx = 6;
		gridy = 28;
		lights = _lights;
		buffer = new byte[(gridy*gridx)*2 + 3];			// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		dashedMode = Boolean.parseBoolean(ConnectionMain.properties.get("dashedLine"));
	}
	
	public int getDefaultDuration(){
		return duration;
	}
	
	public void setDefaultDuration(int millis){
		duration = millis;
	}
	
	public byte[] draw(){
		drawDiagonals();
		drawRows();
		drawColumns();
		drawBlasts();
		checkEverything();
		for(int i=0; i<lights.length; i++){	// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];			// red
			buffer[i*2 + 3] = pair[1];			// blue
		}
		return buffer;
	}
	
	private void checkEverything(){
		Person[] buckets = new Person[lights.length];
		compensation = Float.parseFloat(ConnectionMain.properties.get("forwardCompensation"));	// one and only access per draw
		Iterator <Person> itr = ConnectionMain.personTracker.getPersonIterator();
		while (itr.hasNext()){
			Person personA = itr.next();
			int[] locA = personA.getForwardIntLoc(compensation);
			int elementA = locA[0] + locA[1]*gridx;						// position in the light grid array.
			if(elementA < lights.length && elementA >= 0){				// if this person is in the room.
				if(buckets[elementA] == null){							// if this bucket is empty...
					buckets[elementA] = personA;						// fill it
				} else if(buckets[elementA].id > personA.id){			// if bucket content is newer than this person...
					buckets[elementA] = personA;						// replace with oldest person
				}
				
				drawPerson(elementA);									// draw me.
				checkBlast(personA);									// check to see if a blast should be made
				
				Iterator <Person> itr2 = ConnectionMain.personTracker.getPersonIterator();
				while (itr2.hasNext()){
					Person personB = itr2.next();
					int[] locB = personB.getForwardIntLoc(compensation);					// check location
					int elementB = locB[0] + locB[1]*gridx;									// position in light grid
					if(elementB < lights.length && elementB >= 0){							// if both within the light grid...
						// save a lot of calls by passing these variables
						if(diagonalPairs.size() <= diagPairThreshold){	// if too many diagonals, don't draw new horizontal
							checkRow(personA, locA, elementA, personB, locB, elementB);			// check short ways
						}
						if(rowPairs.size() <= rowPairThreshold){		// if too many horizontals, don't draw new diagonal
							checkDiagonal(personA, locA, elementA, personB, locB, elementB);	// check 45 degree only
						}
					}
				}
			}
		}

		/*
		// THIS IS THE OPTIMIZED VERSION, BUT CAN CAUSE NPE'S
		 
		// get a copy of the current people in the room.  MUST be a copy, since 
		// we need to go back and forth over it many times.  And NO- don't
		// copy a reference to the Collection.  The Collection may return a
		// differen Iterator depending on other threads' work.
		Person[] persons;
		synchronized(ConnectionMain.personTracker.people){
			persons = new Person[ConnectionMain.personTracker.people.size()];
			ConnectionMain.personTracker.people.values().toArray(persons);
		}
		
		Person[] buckets = new Person[lights.length];
		
		for (int outer = 0; outer < persons.length; outer++){	// NOTE: this was person.length-1 but it was skipping people and not DRAWING THEM!

			int[] locA = persons[outer].getForwardIntLoc(compensation);	// get location of the first person.
			int elementA = locA[0] + locA[1]*gridx;						// position in the light grid array.
			// (1) add person to temporary array of buckets if bucket is currently empty

			if(elementA < lights.length && elementA >= 0){				// if this person is in the room.
				if(buckets[elementA] == null){							// if this bucket is empty...
					buckets[elementA] = persons[outer];					// fill it
				} else if(buckets[elementA].id > persons[outer].id){	// if bucket content is newer than this person...
					buckets[elementA] = persons[outer];					// replace with oldest person
				}
				
				drawPerson(elementA);									// draw me.
				checkBlast(persons[outer]);								// check to see if a blast should be made
				
				// for everyone i haven' been pair-checked against, check.
				for (int inner = outer + 1; inner < persons.length; inner++){
					int[] locB = persons[inner].getForwardIntLoc(compensation);					// check location
					int elementB = locB[0] + locB[1]*gridx;									// position in light grid
					if(elementB < lights.length && elementB >= 0){							// if both within the light grid...
						// save a lot of calls by passing these variables
						checkRow(persons[outer], locA, elementA, persons[inner], locB, elementB);			// check short ways
						checkDiagonal(persons[outer], locA, elementA, persons[inner], locB, elementB);	// check 45 degree only
					}
					
				}
			}
		}
		*/
		
		// now we're going to check the columns
		for(int i=0; i<6; i++){
			checkColumn(buckets, i);
		}
	}
	
	
	private void checkColumn(Person[] buckets, int startindex){
		Person lastperson = null;
		int[] lastloc = null;
		int lastdirection = -1;
		for(int i=startindex; i<startindex+(28*5); i+=6){					// for each bucket in column...
			if(buckets[i] != null){										// if bucket is not empty...
				if(lastperson == null){									// if no last person...
					lastperson = buckets[i];								// set this to last person
					lastloc = buckets[i].getForwardIntLoc(compensation);
					lastdirection = buckets[i].getVec();
				} else {																			// if there is a last person...
					int loc[] = buckets[i].getForwardIntLoc(compensation);
					int direction = buckets[i].getVec();											// 0 = up, 1 = down
					if(!columnPairs.containsKey(lastperson.id+","+buckets[i].id) && !columnPairs.containsKey(buckets[i].id+","+lastperson.id)){	// if pair doesn't already exist...
						if((!buckets[i].paired && !lastperson.paired) || !dashedMode){
							if(Math.abs(loc[1] - lastloc[1]) > mindist){									// if greater than minimum distance...
								//System.out.println(direction +" "+ lastdirection);
								if(lastdirection == 1 && direction == 0){			// moving towards each other
									if(Math.random() > 0.5){												// randomize direction of animation
										String id = lastperson.id+","+buckets[i].id;
										columnPairs.put(id, new ColumnPair(id, lastperson, buckets[i]));	// pair with last person
									} else {
										String id = buckets[i].id+","+lastperson.id;
										columnPairs.put(id, new ColumnPair(id, buckets[i], lastperson));	// pair with last person
									}
								} else if(lastdirection == 0 && direction == 1){	// moving away from each other
									if(Math.random() > 0.5){												// randomize direction of animation
										String id = lastperson.id+","+buckets[i].id;
										columnPairs.put(id, new ColumnPair(id, lastperson, buckets[i]));	// pair with last person
									} else {
										String id = buckets[i].id+","+lastperson.id;
										columnPairs.put(id, new ColumnPair(id, buckets[i], lastperson));	// pair with last person
									}
								} else if(lastdirection == 1 && direction == 1){	// both going down
									String id = lastperson.id+","+buckets[i].id;
									columnPairs.put(id, new ColumnPair(id, lastperson, buckets[i]));		// pair with last person
								} else if(lastdirection == 0 && direction == 0){	// both going up
									String id = buckets[i].id+","+lastperson.id;
									//System.out.println(id);
									columnPairs.put(id, new ColumnPair(id, buckets[i], lastperson));		// pair with last person
								}
							}
						}
						if(dashedMode){		// in dashed mode, set last person to null after pairing to leave a gap
							lastperson = null;
						} else {			// non-dashed mode connects people consecutively
							lastperson = buckets[i];
							lastloc = loc;
							lastdirection = direction;
						}
					} else {
						if(dashedMode){		// in dashed mode, set last person to null after pairing to leave a gap
							lastperson = null;
						} else {			// non-dashed mode connects people consecutively
							lastperson = buckets[i];
							lastloc = loc;
							lastdirection = direction;
						}
					}
				}
			}
		}
	}
	
	private void checkRow(Person personA, int[] locA, int elementA, Person personB, int[] locB, int elementB){
		if(locA[1] == locB[1] && Math.abs(locA[0] - locB[0]) > 2){									// if in the same row and atleast 2 lights inbetween...
			if(!rowPairs.containsKey(personA.id+","+personB.id) && !rowPairs.containsKey(personB.id+","+personA.id)){		// if pair doesn't exist...
				rowPairs.put(personA.id+","+personB.id, new HorizontalPair(personA.id+","+personB.id, personA, personB));	// make a new pair
			}
		}
	}
	
	private void checkDiagonal(Person personA, int[] locA, int elementA, Person personB, int[] locB, int elementB){
		// THIS IS THE FASTEST WAY TO DO THIS
		if(!diagonalPairs.containsKey(personA.id+","+personB.id) && !diagonalPairs.containsKey(personB.id+","+personA.id)){
			int diagonalA = -1;
			if(locA[0] == 0){				
				diagonalA = elementA + 35;
			} else if(locA[0] == 5){
				diagonalA = elementA - 35;
			}
	
			if(elementB == diagonalA){		// 45 degree diagonal connection
				diagonalPairs.put(personA.id+","+personB.id, new DiagonalPair(personA.id+","+personB.id, personA, personB));
			}
		}
	}
	
	private void checkBlast(Person person){
		int[] loc = person.getForwardIntLoc(compensation);
		if(loc[1] == 0 && person.yvec > 0 || loc[1] == 27 && person.yvec < 0){	// if within the first row given forward vector...
			if(!person.inside){													// first time inside lighting grid...
				blasts.put(new Integer(blastcounter), new Blast(lights, blasts, blastcounter, person.id, loc[0], person.y));
				blastcounter++;					// BLAST OFF
				person.inside = true;
			}
		}		
	}
	
	
	
	
	
	
	
	private void drawPerson(int element){
		lights[element].setValue(253,0);
	}
	
	private void drawDiagonals(){
		DiagonalPair[] diagonallist = new DiagonalPair[diagonalPairs.size()];	// create empty array
		diagonalPairs.values().toArray(diagonallist);							// populate array with blast objects
		for(int i=0; i<diagonallist.length; i++){
			diagonallist[i].draw();
		}
	}
	
	private void drawRows(){
		HorizontalPair[] rowlist = new HorizontalPair[rowPairs.size()];
		rowPairs.values().toArray(rowlist);
		for(int i=0; i<rowlist.length; i++){
			rowlist[i].draw();
		}
	}
	
	private void drawColumns(){
		ColumnPair[] columnlist = new ColumnPair[columnPairs.size()];
		columnPairs.values().toArray(columnlist);
		for(int i=0; i<columnlist.length; i++){
			columnlist[i].draw();
		}
	}
	
	private void drawBlasts(){
		blastlist = new Blast[blasts.size()];				// create empty array
		blasts.values().toArray(blastlist);					// populate array with blast objects
		for(int i=0; i<blastlist.length; i++){				// for each active blast...
			blastlist[i].move();
			Iterator<Person> p = ConnectionMain.personTracker.getPersonIterator();
			while (p.hasNext()){
				Person person = p.next();
				int[] loc = person.getForwardIntLoc(compensation);	// check location
				if(blastlist[i].gy == loc[1] && blastlist[i].gx == loc[0] && blastlist[i].personID != person.id){
					if(Math.abs(blastlist[i].starty - blastlist[i].y) > blastlist[i].mindist){	// must go atleast minimum distance before exploding
						int fadespeed = 10;
						if(ConnectionMain.personTracker.peopleCount() > 12){
							fadespeed = 15;
						}
						explosions.put(new Integer(explosioncounter), new Explosion(lights, explosioncounter, blastlist[i].gx, blastlist[i].gy, blastlist[i].red, blastlist[i].blue, fadespeed));
						explosioncounter++;
						blasts.remove(new Integer(blastlist[i].id));
					}
				}
			}
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
	}

	
	public void start() {
		// start the global sound
		System.out.println("START: Tracking Main");
		ConnectionMain.properties.put("blueFadeLength", "0.8");
//		try {
//			soundID = ConnectionMain.soundController.newSoundID();
//			ConnectionMain.soundController.globalSound(soundID,soundFile,true,1.0f,10000);
//		} catch (NullPointerException e) {
//			e.printStackTrace();
//		}
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Tracking Main " + soundID);
		reset();
		//ConnectionMain.soundController.killSound(soundID);
	}
	
	public void reset(){
		blasts.clear();
		explosions.clear();
		//pairs.clear();
		blastlist = new Blast[blasts.size()];
		explosionlist = new Explosion[explosions.size()];
		blastcounter = 0;
		explosioncounter = 0;
	}
	
	
	
	
	
	
	
	
	public class ColumnPair{
		
		public String id;
		private Person personA, personB;
		private int[] locA, locB;
		private int[] primaryLoc, secondaryLoc;	// for animation methods
		private int elementA, elementB;
		private int age, state;
		private int column = 0;
		private float drawdistance;
		private float speed;
		private String connectsoundfile;
		private String disconnectsoundfile;
		private boolean broken = false;
		//private int outsidecounter = 0;	// duration a person has been outside the column
		//private int outsidemax = 15;		// max duration before pair disconnects
		
		public ColumnPair(String id, Person personA, Person personB){
			this.id = id;
			this.state = 0;
			this.personA = personA;
			this.personB = personB;
			personA.paired = true;
			personB.paired = true;
			this.locA = personA.getForwardIntLoc(compensation);
			this.locB = personB.getForwardIntLoc(compensation);
			primaryLoc = locA;		
			secondaryLoc = locB;
			this.column = locA[0];
			if(Math.abs(locA[1] - locB[1]) >= 20){												// long sound
				connectsoundfile = ConnectionMain.properties.get("soundColumnConnectionLong");
			} else if(Math.abs(locA[1] - locB[1]) < 20 && Math.abs(locA[1] - locB[1]) > 10){	// medium sound
				connectsoundfile = ConnectionMain.properties.get("soundColumnConnectionMedium");
			} else {																			// short sound
				connectsoundfile = ConnectionMain.properties.get("soundColumnConnectionShort");
			}
			//disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnection");
			drawdistance = 0;
			setSpeed(20,1);		// set line drawing speed
			playConnectSound();
		}
		
		public void draw(){
			age++;
			if(state < 2){
				locA = personA.getForwardIntLoc(compensation);		// check for movement each draw
				locB = personB.getForwardIntLoc(compensation);
				primaryLoc = locA;		// may not be this straight forward
				secondaryLoc = locB;
				elementA = locA[0] + locA[1]*gridx;
				elementB = locB[0] + locB[1]*gridx;
			}
			if(locA[0] != column || locB[0] != column){			// if either has moved out...
				if(age > 15){									// if older than 500ms...
					if(!broken){								// if not broken yet...
						//if(outsidecounter < outsidemax){		// if still counting delay...
						//	outsidecounter++;
						//} else {								// disconnect pair
							state = 2;
							drawdistance = Math.abs(locA[1] - locB[1]);
							if(drawdistance >= 20){
								disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionLong");
							} else if(drawdistance < 20 && drawdistance > 10){
								disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionMedium");
							} else {
								disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionShort");
							}
							broken = true;
							playDisconnectSound();
						//}
					}
				}
			//} else {
				//outsidecounter = 0;								// reset if both are in column
			}
			if(!(elementA < lights.length && elementA >= 0)){
				if(age > 15){
					if(!broken){
						primaryLoc = locB;
						secondaryLoc = locA;
						state = 2;
						drawdistance = Math.abs(locA[1] - locB[1]);
						if(drawdistance >= 20){
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionLong");
						} else if(drawdistance < 20 && drawdistance > 10){
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionMedium");
						} else {
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionShort");
						}
						broken = true;
						playDisconnectSound();
					}
				}
			} else if(!(elementB < lights.length && elementB >= 0)){
				if(age > 15){
					if(!broken){
						primaryLoc = locA;
						secondaryLoc = locB;
						state = 2;
						drawdistance = Math.abs(locA[1] - locB[1]);
						if(drawdistance >= 20){
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionLong");
						} else if(drawdistance < 20 && drawdistance > 10){
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionMedium");
						} else {
							disconnectsoundfile = ConnectionMain.properties.get("soundColumnDisconnectionShort");
						}
						broken = true;
						playDisconnectSound();
					}
				}
			}
			
			// TRYING TO STOP CRAZY CONNECTIONS TO NOWHERE
			if(!ConnectionMain.personTracker.people.containsKey(personA.id)){
				personA.paired = false;
				personB.paired = false;
				columnPairs.remove(id);								// fully remove when line is done disconnecting
			} else if(!ConnectionMain.personTracker.people.containsKey(personB.id)){
				personA.paired = false;
				personB.paired = false;
				columnPairs.remove(id);								// fully remove when line is done disconnecting
			}
			
			if(state == 0){											// connecting
				connect();
			} else if(state == 1){									// connected
				drawLine();
			} else if(state == 2){									// disconnecting
				disconnect();
			}
		}
		
		private void connect(){
			int target = 0;
			if(primaryLoc[1] > secondaryLoc[1]){						// if personA below personB...
				if(drawdistance < primaryLoc[1] - secondaryLoc[1]){		// if distance less than total...
					drawdistance += speed;								// draw distance and speed are always positive
					target = Math.round(primaryLoc[1] - drawdistance);	// nearest light to personA location - distance 
				} else {												// if distance spanned
					target = secondaryLoc[1];							// set target directly to personB location
					state = 1;											// fully connected
				}
				for(int i=primaryLoc[1]; i>target; i--){				// while moving up to target...
					int element = primaryLoc[0] + i*gridx;
					if(element < lights.length && element >= 0){		// if within the light grid...
						lights[element].setBlue(253);
					}
				}
			} else {													// if personA above personB...
				if(drawdistance < secondaryLoc[1] - primaryLoc[1]){		// if distance less than total...
					drawdistance += speed;								// draw distance and speed are always positive
					target = Math.round(primaryLoc[1] + drawdistance);	// nearest light to personA location + distance 
				} else {
					target = secondaryLoc[1];							// set target directly to personB location
					state = 1;											// fully connected
				}
				for(int i=primaryLoc[1]; i<target; i++){				// while moving down to target...
					int element = primaryLoc[0] + i*gridx;				
					if(element < lights.length && element >= 0){		// if within the light grid...
						lights[element].setBlue(253);
					}
				}
			}
		}
		
		public void drawLine(){	
			// BASIC NON-ANIMATED LINE DRAWING
			if(primaryLoc[1] > secondaryLoc[1]){
				for(int i=secondaryLoc[1]+1; i<primaryLoc[1]; i++){
					int element = primaryLoc[0] + i*gridx;
					if(element < lights.length && element >= 0){		// if within the light grid...
						lights[element].setBlue(253);
					}
				}
			} else {
				for(int i=primaryLoc[1]+1; i<secondaryLoc[1]; i++){
					int element = primaryLoc[0] + i*gridx;
					if(element < lights.length && element >= 0){		// if within the light grid...
						lights[element].setBlue(253);
					}
				}
			}
		}
		
		private void disconnect(){
			int target = secondaryLoc[1];
			if(primaryLoc[1] > target){									// if personA below personB...
				if(drawdistance > 0){									// if distance greater than nothing...
					drawdistance -= speed;								// draw distance and speed are always positive
					target = Math.round(primaryLoc[1] - drawdistance);	// nearest light to personA location - distance 
					for(int i=primaryLoc[1]; i>target; i--){			// while moving up to target...
						int element = primaryLoc[0] + i*gridx;
						if(element < lights.length && element >= 0){	// if within the light grid...
							lights[element].setBlue(253);
						}
					}
				} else {
					personA.paired = false;
					personB.paired = false;
					columnPairs.remove(id);								// fully remove when line is done disconnecting
				}
			} else {													// if personA above personB...
				if(drawdistance > 0){									// if distance greater than nothing...
					drawdistance -= speed;								// draw distance and speed are always positive
					target = Math.round(primaryLoc[1] + drawdistance);	// nearest light to personA location + distance
					for(int i=primaryLoc[1]; i<target; i++){			// while moving down to target...
						int element = primaryLoc[0] + i*gridx;				
						if(element < lights.length && element >= 0){	// if within the light grid...
							lights[element].setBlue(253);
						}
					}
				} else {
					personA.paired = false;
					personB.paired = false;
					columnPairs.remove(id);								// fully remove when line is done disconnecting
				}
			}
		}
		
		public void setSpeed(float ms, float distance){
			speed = distance/(ms/30.0f);				// distance divided by frame rate to get there
		}
		
		public void setState(int newstate){
			this.state = newstate;
		}
		
		private void playConnectSound(){
			ConnectionMain.soundController.playSimpleSound(connectsoundfile, primaryLoc[0], primaryLoc[1], 0.8f, "colConnect");
		}
		
		private void playDisconnectSound(){
			ConnectionMain.soundController.playSimpleSound(disconnectsoundfile, secondaryLoc[0], secondaryLoc[1], 0.8f, "colDisonnect");
		}
		
	}
	
	
	
	
	
	
	
	public class HorizontalPair{
		public String id;
		private Person personA, personB;
		private int[] locA, locB;
		private int elementA, elementB, element;
		private int age;
		private float drawdist = 0;
		private int maxdist = 0;
		private int row = 0;
		private boolean fadeout = false;
		private boolean broken = false;
		private String connectsoundfile;
		private String disconnectsoundfile;
		
		public HorizontalPair(String id, Person personA, Person personB){
			this.id = id;
			this.personA = personA;
			this.personB = personB;
			locA = personA.getForwardIntLoc(compensation);
			locB = personB.getForwardIntLoc(compensation);
			row = locA[1];
			elementA = locA[0] + locA[1]*gridx;
			elementB = locB[0] + locB[1]*gridx;
			maxdist = Math.abs(locA[0] - locB[0]);
			connectsoundfile = ConnectionMain.properties.get("soundRowConnection");
			disconnectsoundfile = ConnectionMain.properties.get("soundRowDisconnection");
			playConnectSound();
		}
		
		public void draw(){
			age++;
			locA = personA.getForwardIntLoc(compensation);		// check for movement each draw
			locB = personB.getForwardIntLoc(compensation);
			if((elementA != locA[0] + locA[1]*gridx) || (elementB != locB[0] + locB[1]*gridx)){		// if either has moved out...
				if(age > 30 && drawdist == maxdist){
					if(!broken){	// runs once
						fadeout = true;
						broken = true;
						playDisconnectSound();
					}
				}
			}
			if(drawdist < maxdist && !fadeout){
				drawdist += 0.5;
			} else if(drawdist > 0 && fadeout){
				drawdist -= 0.5;
			} else if(drawdist <= 0 && fadeout){
				diagonalPairs.remove(id);
			}
			
			// TRYING TO STOP CRAZY CONNECTIONS TO NOWHERE
			if(!ConnectionMain.personTracker.people.containsKey(personA.id)){
				rowPairs.remove(id);								// fully remove when line is done disconnecting
			} else if(!ConnectionMain.personTracker.people.containsKey(personB.id)){
				rowPairs.remove(id);								// fully remove when line is done disconnecting
			}
			
			drawLine();
		}
		
		private void drawLine(){
			for(int i=1; i<drawdist; i++){
				if(locA[0] < locB[0]){
					element = locA[0] + row*gridx + i;
				} else {
					element = locA[0] + row*gridx - i;
				}
				if(element < lights.length && element >= 0){
					lights[element].setBlue(253);
				}
			}
		}
		
		private void playConnectSound(){
			ConnectionMain.soundController.playSimpleSound(connectsoundfile, locA[0], locA[1], 0.8f, "rowConnect");
		}
		
		private void playDisconnectSound(){
			ConnectionMain.soundController.playSimpleSound(disconnectsoundfile, locA[0], locA[1], 0.8f, "rowDisconnect");
		}
		
	}
	
	
	
	
	
	
	public class DiagonalPair{
		public String id;
		private Person personA, personB;
		private int[] locA, locB;
		private int elementA, elementB, element;
		private int age;
		private float drawdist = 0;
		private float maxdist = 5;
		private boolean fadeout = false;
		private boolean broken = false;
		private String connectsoundfile;
		private String disconnectsoundfile;
		
		public DiagonalPair(String id, Person personA, Person personB){
			this.id = id;
			this.personA = personA;
			this.personB = personB;
			locA = personA.getForwardIntLoc(compensation);
			locB = personB.getForwardIntLoc(compensation);
			elementA = locA[0] + locA[1]*gridx;
			elementB = locB[0] + locB[1]*gridx;
			connectsoundfile = ConnectionMain.properties.get("soundDiagonalConnection");
			disconnectsoundfile = ConnectionMain.properties.get("soundDiagonalDisconnection");
			playConnectSound();
		}
		
		public void draw(){
			age++;
			locA = personA.getForwardIntLoc(compensation);		// check for movement each draw
			locB = personB.getForwardIntLoc(compensation);
			if(elementA != locA[0] + locA[1]*gridx){			// personA has moved, so disconnect towards personB
				if(age > 30 && drawdist == maxdist){
					if(!broken){
						fadeout = true;
						broken = true;
						playDisconnectSound();
					}
				}
			} else if(elementB != locB[0] + locB[1]*gridx){	// personB has moved, so disconnect towards personA
				if(age > 30 && drawdist == maxdist){
					if(!broken){
						fadeout = true;
						broken = true;
						playDisconnectSound();
					}
				}
			}
			if(drawdist < maxdist && !fadeout){
				drawdist += 0.5;
			} else if(drawdist > 0 && fadeout){
				drawdist -= 0.5;
			} else if(drawdist <= 0 && fadeout){
				diagonalPairs.remove(id);
			}
			
			// TRYING TO STOP CRAZY CONNECTIONS TO NOWHERE
			if(!ConnectionMain.personTracker.people.containsKey(personA.id)){
				diagonalPairs.remove(id);								// fully remove when line is done disconnecting
			} else if(!ConnectionMain.personTracker.people.containsKey(personB.id)){
				diagonalPairs.remove(id);								// fully remove when line is done disconnecting
			}
			
			drawLine();
		}
		
		private void drawLine(){
			for(int i=1; i<drawdist; i++){
				if(locA[0] == 0){
					if(elementA < elementB){
						element = elementA + i*gridx + i;
					} else {
						element = elementA - i*gridx + i;
					}
				} else if(locA[0] == 5){
					if(elementA < elementB){
						element = elementA + i*gridx - i;
					} else {
						element = elementA - i*gridx - i;
					}
				}
				if(element < lights.length && element >= 0){
					lights[element].setBlue(253);
				}
			}
		}
		
		private void playConnectSound(){
			ConnectionMain.soundController.playSimpleSound(connectsoundfile, locA[0], locA[1], 0.8f, "diagConnect");
		}
		
		private void playDisconnectSound(){
			ConnectionMain.soundController.playSimpleSound(disconnectsoundfile, locA[0], locA[1], 0.8f, "diagDisconnect");
		}
		
	}	
	
}

