package net.electroland.connection.animation;

import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;
import java.util.concurrent.ConcurrentHashMap;

public class Chunk {
	public int gridx, gridy;
	public float x,y;
	public float yvel;	
	public int gx, gy, element;
	public int red, blue;
	public int soundID;
	public int id;
	public String soundFile;
	public int[] soundloc;
	public Light[] lights;
	public boolean autoDestroy;
	public ConcurrentHashMap<Integer, Chunk> chunks = new ConcurrentHashMap<Integer, Chunk>();	// active chunks
	
	public Chunk(Light[] _lights){
		lights = _lights;
		gridx = 6;
		gridy = 28;
		soundFile = ConnectionMain.properties.get("soundMatrix");
//		soundID = 0;
		autoDestroy = false;
		randomize();
	}
	
	public Chunk(Light[] _lights, int id, int _gx, float _y, ConcurrentHashMap<Integer, Chunk> _chunks){
		lights = _lights;
		gx = _gx;													// x value in lighting grid
		y = _y;														// normalized y value
		gridx = 6;													// grid dimensions
		gridy = 28;
		autoDestroy = true;
//		soundFile = ConnectionMain.properties.get("soundMatrix");	// sound file
//		soundID = 0;												// sound ID
		if(y < 0.5){												// determine direction of movement
			yvel = (float)(-0.025f*Math.random()) + 0.01f;
		} else {
			yvel = (float)(-0.025f*Math.random()) - 0.01f;
		}
	}
	
	public void move(){
		y += yvel;										// moves up or down
		gy = Math.round(y*gridy);						// position relative to light grid long-ways axis
		// ConnectionMain.soundController.moveSound(soundID, x*3, y);
		if(gy <= 0){
			gy = 0;
			randomize();
		}
		element = gy*gridx + gx;
		if(element < lights.length && element >= 0){
			lights[element].setValue(red, blue);
		}
	}
	
	private void randomize(){
		y = 1;
		yvel = (float)(-0.025f*Math.random()) - 0.01f;
		x = (float)Math.random();
		gx = (int)(Math.random()*6);		// x position in grid
		if(Math.random() > 0.5){
			red = 255;
			blue = (int)(Math.random()*253);
		} else {
			red = (int)(Math.random()*253);
			blue = 255;
		}
		
//		if(soundID == 0){
//			soundID = ConnectionMain.soundController.newSoundID();
//		}
		
//		try {
//			ConnectionMain.soundController.makeSound(soundID, soundFile, false);	// issues make command to max
//			ConnectionMain.soundController.moveSound(soundID, x*3, 0.99f);
//			ConnectionMain.soundController.send("start instance"+soundID);			// issue start command to max
//		} catch (NullPointerException e) {
//			e.printStackTrace();
//		}
	
		
	}
}
