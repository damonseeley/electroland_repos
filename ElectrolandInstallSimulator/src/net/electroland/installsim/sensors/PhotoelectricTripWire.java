package net.electroland.installsim.sensors;

import java.awt.Color;
import java.awt.Graphics;

public class PhotoelectricTripWire {

	public int id;
	public float x;
	public float y;
	public float z;
	public int[] sensingVector;
	
	int bodyWidth = 2;
	int wireStroke = 2;
	Color sensorColor;
	Color wireColor;
	
	boolean tripped = false;
	
	private int value; // 0-255
//	private float uptime;  Unused EGM
	
	//constructor
	public PhotoelectricTripWire(int theid, float locx, float locy, float locz, int[] vector) {
		id = theid; // was forgotten EGM
		x = locx;
		y = locy;
		z = locz;
		sensingVector = vector;
		
		sensorColor = new Color(255, 0, 0);
		wireColor = new Color(0, 0, 255);
		
		value = 0;
	}
	

	public void render(Graphics g) {
		
		//render sensor body
		g.setColor(sensorColor);
		g.fillRect((int)(x-bodyWidth),(int)(y-bodyWidth),bodyWidth*2,bodyWidth*2);
		g.drawString("s" + this.id, (int)x-30, (int)y+2);
		
		//render tripwire
		g.setColor(sensorColor);
		g.drawLine((int)x, (int)y, (int)x+sensingVector[0], (int)y+sensingVector[1]);
		
		
	}

	public void trip() {
		tripped = true;
		// set a timer here for untrip
		// update the timer if tripped again before timer expire
	}
	
	public int getValue() {
		return value;
	}
	
	public String toString() {
		return id + ":(" + x + ", " + y + ") = " + value;
	}


}
