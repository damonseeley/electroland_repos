package net.electroland.enteractive.core;

public class Tile {
	
	private TileController parent;
	private int id, x, y;
	private boolean sensorState;
	private int lightValue;
	private long lastActivated;
	private long turnedOff;
	public boolean rebooting;
	
	public Tile(TileController parent, int id, int x, int y){
		this.parent = parent;
		this.id = id;
		this.x = x;
		this.y = y;
		lastActivated = 0;
		rebooting = false;
		//System.out.println("tile:\t"+id+"\t x: "+x+"\t y:"+y);
	}
	
	public boolean getSensorState(){
		return sensorState;
	}
	
	public void setSensorState(boolean sensorState){
		this.sensorState = sensorState;
		if(sensorState){
			lastActivated = System.currentTimeMillis();
		}
		//System.out.println("sensor state "+sensorState+" tile "+id);
	}
	
	public void reboot(){
		rebooting = true;
		turnedOff = System.currentTimeMillis();
	}
	
	public int getLightValue(){
		return lightValue;
	}
	
	public void setLightValue(int lightValue){
		this.lightValue = lightValue;
		System.out.println("light value "+lightValue+" tile "+id);
	}
	
	public long getAge(){
		return System.currentTimeMillis() - lastActivated;
	}
	
	public long offPeriod(){
		return System.currentTimeMillis() - turnedOff;
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
