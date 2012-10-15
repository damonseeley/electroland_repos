package net.electroland.elvis.blobtracking;


import java.awt.Color;
import java.awt.Graphics2D;
import java.util.HashSet;

public class Blob {
	public int minX = Integer.MAX_VALUE;
	public int maxX = -1;

	public int minY = Integer.MAX_VALUE;
	public int maxY = -1;

	public float centerX = 0;
	public float centerY= 0;

	protected float size = 0;
	
	public int id;

	public static int nextId = 0;
	
	public Blob() {
		id = nextId++;
		nextId %= (Integer.MAX_VALUE - 2);
	}
	
	public Blob(float x, float y, int minX, int maxX, int minY, int maxY, float size) {
		this();
		centerX = x;
		centerY = y;
		this.minX = minX;
		this.maxX = maxX;
		this.minY = minY;
		this.maxY = maxY;
		
		this.size = size;
		
	}
	
	public float getSize() {
		return size;
	}

	
	public float distSqr(float x, float y) {
		float ret = centerX - x;
		ret *= ret;
		float tmp = centerY - y;
		tmp *=tmp;
		ret +=tmp;
		return ret;
	}

	public double dist(float x, float y) {
		return Math.sqrt(distSqr(x,y));
	}
	

	public String toString() {
		return "Blob:" + id + " (" + centerX + ", " +centerY + ")";
	}
	
	public static final int centerDotRadius= 6;
	public static final int centerDotDiameter = centerDotRadius+centerDotRadius;

	public void paint(Graphics2D g) {
		g.setColor(Color.DARK_GRAY);
		g.fillOval((int)centerX - centerDotRadius, (int)centerY - centerDotRadius, centerDotDiameter, centerDotDiameter);
		g.setColor(Color.DARK_GRAY);
		g.drawRect(minX, minY, maxX - minX, maxY - minY);
	}



}
