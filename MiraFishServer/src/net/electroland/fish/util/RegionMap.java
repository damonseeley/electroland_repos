package net.electroland.fish.util;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Polygon;
import java.awt.Rectangle;
import java.awt.image.BufferedImage;


public class RegionMap {
	private int curBitMask = 1;
	
	Bounds bounds;
	BufferedImage image;
	public static BufferedImage CLEAR_IMAGE = null;
	
	public RegionMap(Bounds worldBounds) {
		bounds = new Bounds(worldBounds);
		if(CLEAR_IMAGE==null){
			CLEAR_IMAGE = new BufferedImage((int) worldBounds.getWidth(), (int) worldBounds.getHeight(), BufferedImage.TYPE_INT_ARGB);			
			for(int x = 0 ; x < CLEAR_IMAGE.getWidth();x++) {
				for(int y = 0; y < CLEAR_IMAGE.getHeight();y++) {
					CLEAR_IMAGE.setRGB(x, y, 0);
				}
			}
			
		}
 		image = new BufferedImage((int) worldBounds.getWidth(), (int) worldBounds.getHeight(), BufferedImage.TYPE_INT_ARGB);

		clearImage(image);
	}
	

	/** this method inefficent only call at startup 
	 * returns a bitmask for later comparison
	 * */

	public int addRegion(BufferedImage bi) {
		if (curBitMask == 0) return 0; // out of bits
		
		int curColor = curBitMask;
		curBitMask = curBitMask <<  1; //inc bitmask


		for(int x = 0; x < bi.getWidth(); x++) {
			for(int y = 0; y < bi.getHeight(); y++) {
				if(bi.getRGB(x, y) != 0) {
					System.out.println("oring" + x + ", " + y + "  " + image.getRGB(x, y) +"|" +curColor
							+ "= " + (image.getRGB(x, y) | curColor));
					int pix =  image.getRGB(x, y);
					image.setRGB(x, y, pix | curColor);
				}
			}
		}
		return curColor;
		
	}
	
	public int addRegion(Rectangle r) {
		System.out.println("about to create image");
		BufferedImage tmpImage =  new BufferedImage((int) bounds.getWidth(), (int) bounds.getHeight(), BufferedImage.TYPE_INT_ARGB);
		System.out.println("image created");

		Graphics2D g2d = tmpImage.createGraphics();
		clearImage(tmpImage);
		g2d.setColor(Color.WHITE);
		g2d.fillRect(r.x, r.y, r.width, r.height);
		return addRegion(tmpImage);
		
		
	}
	public int addRegion(Polygon p) {

		BufferedImage tmpImage =  new BufferedImage((int) bounds.getWidth(), (int) bounds.getHeight(), BufferedImage.TYPE_INT_ARGB);
		Graphics2D g2d = tmpImage.createGraphics();
		clearImage(tmpImage);
		g2d.setColor(Color.WHITE);
		g2d.fillPolygon(p);

		return addRegion(tmpImage);


	}
	
	public int getRegions(float x, float y) {
		return image.getRGB((int)x, (int)y);
	}
	
	public boolean isInRegion(float x, float y, int bitmask) {
		return (image.getRGB((int)x,(int) y) & bitmask) != 0;
	}
	
	
	public static void clearImage(BufferedImage bi) { // there must be a better way
		for(int x = 0 ; x < bi.getWidth();x++) {
			for(int y = 0; y < bi.getHeight();y++) {
				bi.setRGB(x, y, 0);
			}
		}
	}
	
	public static void main(String args[]) {
		Bounds bounds = new Bounds(0,0,1200, 1600,0,0);
		RegionMap rm = new RegionMap(bounds);
		BufferedImage tmpImage =  new BufferedImage((int) bounds.getWidth(), (int) bounds.getHeight(), BufferedImage.TYPE_INT_ARGB);
		clearImage(tmpImage);
		
	
		Graphics2D g2d = tmpImage.createGraphics();

		g2d.setColor(Color.WHITE);
		g2d.fillRect(100, 100, 400, 800);
		int r1 = rm.addRegion(tmpImage);
		clearImage(tmpImage);
		
		g2d.setColor(Color.WHITE);
		g2d.fillRect(200, 200, 800, 800);
		int r2 = rm.addRegion(tmpImage);
		
//		System.out.println("r1:" + r1 + "  r2:"+r2);
		
		int x = 10;
		int y = 10;
		int val = rm.getRegions(x,y);
		System.out.println("("+x+ ","+y+")  in R1:"+ ((val&r1)!=0)+"    in R2:" + ((val&r2)!=0));
		
		x=150;
		y=150;
		val = rm.getRegions(x,y);
		System.out.println("("+x+ ","+y+")  in R1:"+ ((val&r1)!=0)+"    in R2:" + ((val&r2)!=0));
		
		x=150;
		y=150;
		val = rm.getRegions(x,y);
		System.out.println("("+x+ ","+y+")  in R1:"+ ((val&r1)!=0)+"    in R2:" + ((val&r2)!=0));
		
		x=300;
		y=300;
		val = rm.getRegions(x,y);
		System.out.println("("+x+ ","+y+")  in R1:"+ ((val&r1)!=0)+"    in R2:" + ((val&r2)!=0));
		
		
	}
	

}
