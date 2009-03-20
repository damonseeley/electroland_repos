package net.electroland.enteractive.core;

/**
 * Triggers enter/exit events and stores age/history data.
 * @author asiegel
 */

public class Person {
	
	private int id, linearLoc, age, x, y;

	public Person(int id, int linearLoc, int x, int y){
		this.id = id;
		this.linearLoc = linearLoc;
		this.x = x;
		this.y = y;
	}
	
	public int getID(){
		return id;
	}
	
	public int getAge(){
		return age;
	}
	
	public void moveTo(int x, int y){
		this.x = x;
		this.y = y;
	}
	
	public int[] getLoc(){
		int[] loc = new int[2];
		loc[0] = x;
		loc[1] = y;
		return loc;
	}
	
	public int getLinearLoc(){
		return linearLoc;
	}
	
}
