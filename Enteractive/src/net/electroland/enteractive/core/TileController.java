package net.electroland.enteractive.core;

import java.util.ArrayList;
import java.util.List;

public class TileController {
	
	private String ip;
	private int id, start, end;
	private List<Tile> tiles;
	
	public TileController(int id, String ip, int start, int end){
		this.id = id;
		this.ip = ip;
		this.start = start;
		this.end = end;
		tiles = new ArrayList<Tile>();
	}
	
	public void addTile(Tile tile){
		tiles.add(tile);
	}

	public String getAddress(){
		return ip;
	}
	
	public int getID(){
		return id;
	}
	
	public int getOffset(){
		return start;
	}

}
