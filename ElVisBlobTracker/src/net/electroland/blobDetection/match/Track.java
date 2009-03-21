package net.electroland.blobDetection.match;

import java.awt.Color;
import java.awt.Graphics2D;

import net.electroland.blobDetection.Blob;


public class Track {
	public float x;
	public float y;	
	int id;
	String idStr = null;
	public static int nextId = 0;

	boolean isProvisional = true;
	boolean isRemoved = false;

	int certianCnt = 0;
	int deleteCnt = 0;

	public  int framesUntilCertian;
	public  int frameUntilDeleted;



	public Track(int framesUntilCertian, int frameUntilDeleted) {
		id= nextId++;
		this.framesUntilCertian = framesUntilCertian;
		this.frameUntilDeleted = frameUntilDeleted;
		certianCnt = framesUntilCertian;
		deleteCnt = frameUntilDeleted;
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

	public void setBlobLoc(Blob b) {
		if(b == CSP.UNMATCHED) {
			deleteCnt--;
			if(deleteCnt <= 0) {
				isRemoved = true;
			}
		} else {
//			System.out.println(certianCnt);
			isProvisional = (certianCnt-- >= 0);
			deleteCnt = frameUntilDeleted;
			// might want to cal velocity in future of look at color or size to improve matchs
			x = b.centerX;
			y = b.centerY;
		}


	}
}
