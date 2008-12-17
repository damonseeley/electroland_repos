package net.electroland.connection.core;

/**
 * Adapted from CoopLights Light.java authored by Eitan Mendelowitz.
 * Revised by Aaron Siegel
 */

public class Light {
	public int id;
	public int x, y;
	public static final int MAX_BRIGHT = 253;
	public float redfade = Float.parseFloat(ConnectionMain.properties.get("redFadeLength"));
	public float bluefade = Float.parseFloat(ConnectionMain.properties.get("blueFadeLength"));
	private boolean newred = false;
	private boolean newblue = false;
	public int red;
	public int blue;
	public boolean redIncrease = false;
	public boolean blueIncrease = false;
	public boolean fading = false;		// fading or not
	public boolean fadingin = false;		// inward
	public boolean fadingout = false;	// outward
	public boolean holding = false;
	public int fadeInLength, holdLength, fadeOutLength;
	public int fadecount, fadespeed, holdcount;
	
	public Light(int theid, int _x, int _y) {
		id = theid;
		x = _x;
		y = _y;
		red = 0;
		blue = 0;
	}
	
	public int[] getValue(){
		return new int[] {red,blue};
	}
	
	public byte[] process(){
		if(fading){	// fading mode causes light to throb
			if(fadingin){			// fading in
				if(red+fadespeed <= 250){
					red += fadespeed;
				} else {
					red = 250;
					fadingin = false;
				}
			} else if(fadingout){	// fading out
				if(red-fadespeed >= 0){
					red -= fadespeed;
				} else {
					red = 0;
					fadingout = false;
					//fading = false;	// non looping
					holding = true;		// looping
				}
			} else if(holding){	// holding while off
				if(holdcount < holdLength){
					holdcount++;
				} else {
					holdcount = 0;
					fadespeed = (int)(250/((fadeInLength/1000.0f)*30));
					holding = false;
					fadingin = true;
				}
			} else {				// holding well lit
				if(holdcount < holdLength){
					holdcount++;
				} else {
					holdcount = 0;
					fadespeed = (int)(250/((fadeOutLength/1000.0f)*30));
					fadingout = true;
				}
			}
		} else {
			if(!newred){
				if(red > 1){
					red = (int)(red * Float.parseFloat(ConnectionMain.properties.get("redFadeLength")));	// dampens color
				} else {
					red = 0;
				}
			} else {
				newred = false;
			}
			if(!newblue){
				if(blue > 1){
					blue = (int)(blue * Float.parseFloat(ConnectionMain.properties.get("blueFadeLength")));	// dampens color
				} else {
					blue = 0;
				}
			} else {
				newblue = false;
			}
		}
		return new byte[] {(byte)red,(byte)blue};
	}

	
	public void setBlue(int newvalue){
		if (newvalue > MAX_BRIGHT) {		// BLUE
			blue = MAX_BRIGHT;
		} else if (newvalue < 0) {
			blue = 0;
		} else {
			blue = newvalue;
		}
		newblue = true;
	}
	
	public void setRed(int newvalue){
		if (newvalue > MAX_BRIGHT) {		// RED
			red = MAX_BRIGHT;
		} else if (newvalue < 0) {
			red = 0;
		} else {
			red = newvalue;
		}
		newred = true;
	}
	
	public void setValue(int newred, int newblue) {	// RED and BLUE version
		setRed(newred);
		setBlue(newblue);
	}
	
	public void setValue(int newcolor) {				// PURPLE version
		setRed(newcolor);
		setBlue(newcolor);
	}
	
	public void addValue(int newred, int newblue){
		if(newred > red){
			setRed(newred);
		}
		if(newblue > blue){
			setBlue(newblue);
		}
	}
	
	public void startFade(int fadein, int hold, int fadeout){
		fadeInLength = fadein;
		holdLength = (int)((hold/1000.0f)*30);
		fadeOutLength = fadeout;
		fadespeed = (int)(250/((fadeInLength/1000.0f)*30));
		fadingin = true;
		fading = true;
	}
	
}
