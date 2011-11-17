package net.electroland.installsim.core;
import java.awt.Color;
import java.awt.Font;
import java.awt.Graphics;
import java.awt.geom.Point2D;
import java.util.Random;
import java.util.Vector;


public class Person {
	
	public Integer id;
	public float x;
	public float y;
	public float z;
	public float speed;
	public long birthdate;
	public Color color;
	public float size;
	
	public static float SMOOTHING_RATE = .8f; // the more noise is eliminated (but the more someone has to stand still to notice)
	private static float STILL_THRESH_SQR = 14.0f; // the distance underwhich movment is considered not moving

	public boolean isStill = false;
	private float vector[];
	


	public Person(int idvalue, float xvalue, float yvalue, float zvalue, float speed) {
		id = new Integer(idvalue); //Integer object is used by hashmap so we might as well store id as an object
		x = xvalue;
		y = yvalue;
		z = zvalue;
		this.speed = speed;
		birthdate = System.currentTimeMillis();
		
		Random r = new Random();
		color = new Color(r.nextInt(255),r.nextInt(255),r.nextInt(255));
		size = 10;
	}
	
	public void render(Graphics g, int id) {
		
		g.setColor(color);
		g.fillOval((int)(x-size/2), (int)(y-size/2), (int)size, (int)size);
		
		Color c = new Color(128,128,128);
		g.setColor(c);
		Font font = new Font("Arial", Font.PLAIN, 7);
	    g.setFont(font);
		g.drawString("ID:" + id, (int)(x-8), (int)(y-6));
		//g2.drawString((int)p.x + ", " + (int)p.y + ", " + (int)p.z, (int)personCircle.x-10, (int)personCircle.y+h*3+1);
		
		
	}
	
	public void setVector(float[] vec) {
		vector = vec;
	}
	
	public float[] getVec () {
		return vector;
	}
	

	public void setLoc(float xvalue, float yvalue){
		x = xvalue;
		y = yvalue;	
	}
	
	public void standingStill() {
		isStill = true;		
	}
	public void moving(){
		isStill = false;
	}
	
	public long getAge(){
		long myAge = System.currentTimeMillis() - birthdate;
		return myAge;
	}
	
	
	
	// helps with debugging
	public String toString() {
		return "p" + id + ":(" + x + ", " + y + ", " + z + ")";
	}



}
