package net.electroland.blobDetection;

import java.awt.Color;
import java.awt.Graphics2D;
import java.util.HashSet;

public class Blob {
	boolean centerIsCalculated = false;

	HashSet<Integer> ids = new HashSet<Integer>();

	public int minX = Integer.MAX_VALUE;
	public int maxX = -1;

	public int minY = Integer.MAX_VALUE;
	public int maxY = -1;

	// these are not valid until calcCenter is called 	
	public float centerX = 0;
	public float centerY= 0;

	protected int size = 0;
	
	public int id;
	public static int nextId = 0;

	public Blob() {
		id = nextId++;
	}
	
	public Blob(float x, float y) {
		this();
		centerX = x;
		centerY = y;
	}
	
	public int getSize() {
		return size;
	}

	public void addPoint(int x, int y) {

		
		long beforeSize = size;

		centerX += x;
		centerY += y;

		size++;		

		minX = (x < minX) ? x : minX;
		minY = (y < minY) ? y : minY;

		maxX = (x > maxX) ? x : maxX;
		maxY = (y > maxY) ? y : maxY;




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
	public void calcCenter() {
		
		float scaler = 1.0f / (float) size;
		centerX *= scaler;
		centerY *= scaler;
		centerIsCalculated = true;
		

	}

	/*
	 * changes to represent merger of this object and the other blob
	 * should not be used if calcCenter has been called on either object
	 */
	public void merger(Blob blob) {
		ids.addAll(blob.ids);

		centerX += blob.centerX;
		centerY += blob.centerY;
		
		

		size += blob.size;
		
		//System.out.println(" = " + size);

		minX = (blob.minX < minX) ? blob.minX : minX;
		minY = (blob.minY < minY) ? blob.minY : minY;

		maxX = (blob.maxX > maxX) ? blob.maxX : maxX;
		maxY = (blob.maxY > maxY) ? blob.maxY : maxY;

	}

	public String toString() {
		return "Blob:" + id + " (" + centerX + ", " +centerY + ")";
	}
	public static final int centerDotRadius= 6;
	public static final int centerDotDiameter = centerDotRadius+centerDotRadius;

	public void paint(Graphics2D g) {
		g.setColor(Color.BLUE);
		g.fillOval((int)centerX - centerDotRadius, (int)centerY - centerDotRadius, centerDotDiameter, centerDotDiameter);
		g.setColor(Color.RED);
		g.drawRect(minX, minY, maxX - minX, maxY - minY);
	}



}
