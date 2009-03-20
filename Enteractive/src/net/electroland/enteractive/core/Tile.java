package net.electroland.enteractive.core;

public class Tile {
	
	private TileController parent;
	private int id, x, y;
	private boolean sensorState;
	private int lightValue;
	
	public Tile(TileController parent, int id, int x, int y){
		this.parent = parent;
		this.id = id;
		this.x = x;
		this.y = y;
		//System.out.println("tile:\t"+id+"\t x: "+x+"\t y:"+y);
	}
	
	public boolean getSensorState(){
		return sensorState;
	}
	
	public void setSensorState(boolean sensorState){
		this.sensorState = sensorState;
		//System.out.println("sensor state "+sensorState+" tile "+id);
	}
	
	public int getLightValue(){
		return lightValue;
	}
	
	public void setLightValue(int lightValue){
		this.lightValue = lightValue;
		System.out.println("light value "+lightValue+" tile "+id);
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
