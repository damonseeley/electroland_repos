package net.electroland.coopLights.core;

public class Light {

	public int id;
	public float x;
	public float y;
	
	public static final int MAX_BRIGHT = 255;
	
	private int value; // 0-255
//	private float uptime;  Unused EGM
	
	//constructor
	public Light(int theid, float locx, float locy) {
		id = theid; // was forgotten EGM
		x = locx;
		y = locy;
		value = 0;
	//	uptime = 0;
	}

	
	public void setValue(int newbright) {
		if (newbright > 255) {
			value = 255;
		} else if (newbright < 0) {
			value = 0;
		} else {
			value = newbright;
		}
		if (value > 0) {
			//start timer here for max on time calcs
		}
	}
	
	public void incValue(int incAmount){
		if (value + incAmount >255) {
			value = 255;
		} else if (value + incAmount < 0){
			//value = 0;
		}else{
			value+=incAmount;
		}
	}
	
	public void decValue(int decAmount){
		if (value - decAmount >255) {
			value = 255;
		} else if (value - decAmount < 0){
			value = 0;
		}else{
			value-=decAmount;
		}
	}
	
	public int getValue() {
		return value;
		
	}
	
	public String toString() {
		return id + ":(" + x + ", " + y + ") = " + value;
	}


}
