package net.electroland.connection.animation;

import java.util.concurrent.ConcurrentHashMap;
import net.electroland.connection.core.ConnectionMain;
import net.electroland.connection.core.Light;

public class Blast{
	
	public Light[] lights;
	public ConcurrentHashMap<Integer, Blast> blasts;
	public int gridx, gridy;
	public float x, y, dist, starty;
	public float yvel;	
	public int gx, gy, element;
	public int red, blue;
	public int id;
	public int personID;
	public float mindist = 0.3f;
	public String soundFile;
	
	public Blast(Light[] _lights, ConcurrentHashMap<Integer, Blast> _blasts, int _id, int _personID, int _gx, float _y){
		gridx = 6;
		gridy = 28;
		lights = _lights;
		blasts = _blasts;
		id = _id;
		personID = _personID;
		gx = _gx;
		y = _y;
		gy = Math.round(y*gridy);
		soundFile = ConnectionMain.properties.get("soundBlast");
		if(Math.random() > 0.5){									// limited range of purples
			red = 255;
			blue = (int)(Math.random()*100)+153;
		} else {
			red = (int)(Math.random()*100)+153;
			blue = 255;
		}
		starty = y;
		dist = 0;
		
		if(y < 0.5){												// determine direction of movement
			yvel = (float)(0.025f*Math.random()) + 0.01f;
		} else {
			yvel = (float)(-0.025f*Math.random()) - 0.01f;
		}
		
		ConnectionMain.soundController.playSimpleSound(soundFile, gx, gy, 0.8f, "shooter");
	}
	
	public void move(){
		y += yvel;										// moves up or down
		x = gx/(float)6;
		gy = Math.round(y*gridy);						// position relative to light grid long-ways axis
		element = gy*gridx + gx;
		if(element < lights.length && element >= 0){
			lights[element].setValue(red, blue);
		} else if(element > lights.length && yvel > 0){
			blasts.remove(new Integer(id));
		} else if(element < 0 && yvel < 0){
			blasts.remove(new Integer(id));
		}
	}
}
