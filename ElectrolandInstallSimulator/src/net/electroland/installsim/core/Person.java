package net.electroland.installsim.core;
import java.awt.Color;
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
	
	public static float SMOOTHING_RATE = .8f; // the more noise is eliminated (but the more someone has to stand still to notice)
	private static float STILL_THRESH_SQR = 14.0f; // the distance underwhich movment is considered not moving

	public boolean isStill = false;
	private float vector[];

	public Person(int idvalue, float xvalue, float yvalue, float zvalue) {
		id = new Integer(idvalue); //Integer object is used by hashmap so we might as well store id as an object
		x = xvalue;
		y = yvalue;
		z = zvalue;


		birthdate = System.currentTimeMillis();
		
		Random r = new Random();
		color = new Color(r.nextInt(255),r.nextInt(255),r.nextInt(255));
	
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
		


		
		/*if (smoothDistSqr < STILL_THRESH_SQR) {
			standingStill();
		} else {
			moving();
		}*/
				
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
