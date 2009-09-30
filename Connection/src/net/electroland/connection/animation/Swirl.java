package net.electroland.connection.animation;

import java.util.concurrent.ConcurrentHashMap;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Swirl {

	public Light[] lights;
	public byte[] buffer, pair;
	public int gridx, gridy;
	//public Ticker[] tickers;
	public int framesPerRefresh;
	public int frameCounter;
	public ConcurrentHashMap<Integer, Ticker> tickers = new ConcurrentHashMap<Integer, Ticker>();
	public Ticker[] tickerlist;
	public int tickercounter;
	public int mandatorywait = 2;
	public int waitcounter = 0;
	public int soundID;
	public String soundFile;
	
	public Swirl(Light[] _lights){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		framesPerRefresh = 1;
		frameCounter = 0;
		buffer = new byte[(28*6)*2 + 3];					// allocate the packet and set it's start, cmd, and end bytes
		buffer[0] = (byte)255; 								// start byte
		buffer[1] = (byte)0;								// command byte
		buffer[buffer.length-1] = (byte)254; 				// end byte
		soundFile = ConnectionMain.properties.get("soundSwirlGlobal");
		soundID = -1;
		//tickers = new Ticker[7];
		/*
		for(int i=0; i<tickers.length; i++){
			// positions need to be spaced out, not random
			tickers[i] = new Ticker((int)(Math.random()*4)+1, (int)(Math.random()*26)+1);
		}
		*/
		//reset();
	}
	
	public void start() {
		// start the global sound
		System.out.println("START: Swirl");
		soundID = ConnectionMain.soundController.newSoundID();
		ConnectionMain.soundController.globalSound(soundID,soundFile,true,1.0f,10000,"swirl");
	}
	
	public void stop() {
		// stop the global sound
		System.out.println("STOP: Swirl " + soundID);
		//ConnectionMain.soundController.killSound(soundID);
	}
	
	public void reset(){
		/*
		tickers[0] = new Ticker((int)(Math.random()*4)+1, 1);
		tickers[1] = new Ticker((int)(Math.random()*4)+1, 5);
		tickers[2] = new Ticker((int)(Math.random()*4)+1, 9);
		tickers[3] = new Ticker((int)(Math.random()*4)+1, 13);
		tickers[4] = new Ticker((int)(Math.random()*4)+1, 17);
		tickers[5] = new Ticker((int)(Math.random()*4)+1, 21);
		tickers[6] = new Ticker((int)(Math.random()*4)+1, 25);
		*/
	}
	
	public byte[] draw(){
		if(Math.random() > 0.65){						// greater than 1 in 5 chance of a new ticker
			tickers.put(new Integer(tickercounter), new Ticker(tickercounter, (int)(Math.random()*5)+1, (int)(Math.random()*26)+1));
			tickercounter++;
			//System.out.println("NEW Ticker");
		}
		if(frameCounter == framesPerRefresh){
			tickerlist = new Ticker[tickers.size()];
			tickers.values().toArray(tickerlist);
			for(int i=0; i<tickerlist.length; i++){
				tickerlist[i].move();
			}
			/*
			for(int i=0; i<tickers.length; i++){
				tickers[i].move();
			}
			*/
			
			frameCounter = 0;
		} else {
			frameCounter++;
		}
		for(int i=0; i<lights.length; i++){				// process each light
			pair = lights[i].process();
			buffer[i*2 + 2] = pair[0];						// red
			buffer[i*2 + 3] = pair[1];						// blue
		}
		return buffer;
	}
	
	
	
	
	public class Ticker{
		public int x, y, position;
		public float red, blue;
		public int[] positions;
		public boolean clockwise;
		public int soundID;
		public String soundFile;
		public String[] soundFileList;
		public int[] soundloc;
		public int gain = 1;
		public int id;
		public int age = 253;
		public int brightness = 253;
		
		public Ticker(int _id, int _x, int _y){
			id = _id;
			x = _x;
			y = _y;
			clockwise = true;
			if(Math.random() > 0.5){
				//red = 255;
				//blue = (int)(Math.random()*255);
				red = 1;
				blue = (float)Math.random();
			} else {
				//red = (int)(Math.random()*255);
				//blue = 255;
				red = (float)Math.random();
				blue = 1;
			}
			
			soundFileList = new String[3];
			soundFileList[0] = ConnectionMain.properties.get("soundSwirlA");
			soundFileList[1] = ConnectionMain.properties.get("soundSwirlB");
			soundFileList[2] = ConnectionMain.properties.get("soundSwirlC");
			soundFile = ConnectionMain.soundController.randomSound(soundFileList);
			//soundFile = ConnectionMain.properties.get("soundSwirl");
			//soundloc = ConnectionMain.soundController.getNearestSpeaker(x,y);
			//System.out.println("NEAREST SPKR: "+soundloc[0] + " " + soundloc[1] + "   " + x + "," + y);
			// only runs once when instantiated
//			try{
//				soundID = ConnectionMain.soundController.newSoundID();	// JUST RETURNS ID
//				ConnectionMain.soundController.send("simple instance"+soundID+" "+soundFile+" "+soundloc[0]+" "+soundloc[1]+" 0 "+gain);
//			} catch(NullPointerException e){
//				System.err.println("sound output error: "+e);
//			}
			definePositions();
		}
		
		private void definePositions(){
			// this is for a 9 light square
			/*
			positions = new int[8];
			positions[0] = (y-1)*gridx + x-1;	// upper left
			positions[1] = (y-1)*gridx + x;		// top
			positions[2] = (y-1)*gridx + x+1;	// upper right
			positions[3] = y*gridx + x+1;		// right
			positions[4] = (y+1)*gridx + x+1;	// lower right
			positions[5] = (y+1)*gridx + x;		// below
			positions[6] = (y+1)*gridx + x-1;	// lower left
			positions[7] = y*gridx + x-1;		// left
			*/
			// this is for a 4 light square
			
			positions = new int[4];
			positions[0] = (y-1)*gridx + x-1;	// upper left
			positions[1] = (y-1)*gridx + x;		// top
			positions[2] = y*gridx + x;			// home
			positions[3] = y*gridx + x-1;		// left
			
		}
		
		public void move(){
			// ticker goes in circles around element
			if(position == positions.length-1){
				position = 0;
			} else {
				position++;
			}
			lights[positions[position]].setValue((int)(red*brightness), (int)(blue*brightness));
			age -= 10;
			if(age < 128){
				brightness -= 10;
			}
			if(age <= 0){
				tickers.remove(new Integer(id));
			}
		}
	}
}
