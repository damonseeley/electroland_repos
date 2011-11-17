package net.electroland.installsim.core;

import java.awt.Graphics;
import java.util.Vector;
import java.util.concurrent.ConcurrentHashMap;

import net.electroland.installsim.sensors.Sensor;


public class ModelGeneric {
	
	public ConcurrentHashMap<Integer, Person> people;
	public Vector<Sensor>sensors;

	public static float xScale;
	public static float yScale;
	public static float xOffset;
	public static float yOffset;

	public static Integer DUMMYID = -1; 
	
	public ModelGeneric() {

	}
	

	public void initPeople(){
	
			
	}
	
	public void updatePeople(){
	
	}
	
	public void render(Graphics g) {
	
	}



}
