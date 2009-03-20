package net.electroland.enteractive.core;

import java.util.ArrayList;
import java.util.List;

public class TileController {
	
	private String ip;
	private int id, start, end;
	private List<Tile> tiles;
	private int gridWidth = 16;
	
	public TileController(int id, String ip, int start, int end){
		this.id = id;
		this.ip = ip;
		this.start = start;
		this.end = end;
		//System.out.println(id+" "+ip+" "+start+" "+end);
		tiles = new ArrayList<Tile>();
		for(int i=start; i<=end; i++){
			int x = i%gridWidth;
			int y = i/gridWidth;
			if(x == 0){		// compensation for modulo resulting in 0
				x = 16;
			} else {
				y++;
			}
			tiles.add(new Tile(this, i, x, y));
		}
	}
	
	public void addTile(Tile tile){
		tiles.add(tile);
	}
	
	public List<Tile> getTiles(){
		return tiles;
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
	
	public int getLength(){
		return end-(start-1);
	}

}
