package net.electroland.enteractive.core;

public class Tile {
	
	private int id, x, y;
	
	public Tile(int id, int x, int y){
		this.id = id;
		this.x = x;
		this.y = y;
	}
	
	public int getID(){
		return id;
	}
	
	public void getRasterPosition(){
		// TODO return the x/y pixel position of the corresponding detector
	}

}
