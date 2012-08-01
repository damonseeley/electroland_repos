package net.electroland.fish.util;

public class Content {
	
	public int id;
	public String name;
	public int width;
	public int height;
	public long duration;
	
	public float halfHeight;
	public float halfWidth;

	
	public Content() {
	}
	public Content(int id, String name, int w, int h, long d) {
		this.id = id;
		this.name = name;
		width = w;
		height = h;
		duration = d;
		
		updateHW();
		
	}
	
	public void updateHW() {
		halfHeight = ((float)height)*.5f;
		halfWidth = ((float)width) * .5f;
		
	}

	public Content(Content c) {
		this.id = c.id;
		this.name = c.name;
		width = c.width;
		height = c.height;
		duration = c.duration;
		halfHeight = c.halfHeight;
		halfWidth = c.halfWidth;		
	}
}
