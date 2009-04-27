package net.electroland.blobDetection.match;

import java.awt.Color;
import java.awt.Graphics2D;

import net.electroland.blobDetection.Blob;


public class Track {
	public float vx=0;
	public float vy=0;

	public float x;
	public float y;
	public float width, height;
	public int id;
	String idStr = null;
	public static int nextId = 0;

	boolean isProvisional = true;
	boolean isRemoved = false;

	int certianCnt = 0;
	int deleteCnt = 0;

	public  int framesUntilCertian;
	public  int frameUntilDeleted;

	public float velocityMatchPercentage;

	public Track(int framesUntilCertian, int frameUntilDeleted, float velocityMatchPercentage) {
		id= nextId++;
		this.framesUntilCertian = framesUntilCertian;
		this.frameUntilDeleted = frameUntilDeleted;
		certianCnt = framesUntilCertian;
		deleteCnt = frameUntilDeleted;
		this.velocityMatchPercentage = velocityMatchPercentage;
	}

	public String toString() {
		return "Track:" + id + " (" + x +" ," + y +")"; 
	}

	public static final Color PROV_COLOR = new Color(55,0,0);

	public static final int centerDotRadius= 3;
	public static final int centerDotDiameter = centerDotRadius+centerDotRadius;

	public void paint(Graphics2D g) {
		if(idStr == null) {
			idStr = Integer.toString(id);
		}
		if(isProvisional) {
			g.setColor(PROV_COLOR);
		} else {
			g.setColor(Color.RED);
		}
		g.drawString(idStr, x+centerDotRadius, y+centerDotRadius);
		g.fillOval((int)x - centerDotRadius, (int)y - centerDotRadius, centerDotDiameter, centerDotDiameter);
		
	}

	public void move() {
		x += vx;
		y += vy;
	}

	public void setBlobLoc(Blob b) {
		if(b == CSP.UNMATCHED) {
			deleteCnt--;
			if(deleteCnt <= 0) {
				isRemoved = true;
			}
		} else {
			
			// if velocityMatchPercentage and the track has been around for a frame
			// calculate velocity using a running average
			if((velocityMatchPercentage >= 0) && (certianCnt < framesUntilCertian)){
				
				vx *= velocityMatchPercentage;
				vx += (1-velocityMatchPercentage) * (b.centerX - x);
				
				vy *= velocityMatchPercentage;
				vy += (1-velocityMatchPercentage) * (b.centerY - y);
				
				
			}
			
			isProvisional = (certianCnt-- >= 0);
			deleteCnt = frameUntilDeleted;

			
			
			x = b.centerX;
			y = b.centerY;
			height = b.maxY - b.minY;
			width = b.maxX - b.minX;
		}


	}

	public boolean isProvisional() {
		return isProvisional;
	}
}
