package net.electroland.enteractive.core;

public class Tile {
	
	private TileController parent;
	private int id, x, y;
	
	public Tile(TileController parent, int id, int x, int y){
		this.parent = parent;
		this.id = id;
		this.x = x;
		this.y = y;
		//System.out.println("tile:\t"+id+"\t x: "+x+"\t y:"+y);
	}
	
	public int getID(){
		return id;
	}
	
	public TileController getController(){
		return parent;
	}
	
	public int getX(){
		return x;
	}
	
	public int getY(){
		return y;
	}
	
	public void getRasterPosition(){
		// TODO return the x/y pixel position of the corresponding detector
	}

}
