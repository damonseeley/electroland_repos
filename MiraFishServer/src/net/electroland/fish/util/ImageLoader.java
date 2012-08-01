package net.electroland.fish.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;

import javax.imageio.ImageIO;
import javax.vecmath.Vector3f;

public class ImageLoader {

	
	public static boolean[][] getBoolArrayForImageFile(String fileName) {
		BufferedImage bi;
		try {
			bi = ImageIO.read(new File(fileName));
		} catch (IOException e) {
			System.out.println("can't read " + fileName);
			e.printStackTrace();
			return null;
		}
		boolean[][] returnVal = new boolean[bi.getWidth()][bi.getHeight()];
		for(int x = 0; x < bi.getWidth();x++) {
			for(int y = 0; y < bi.getHeight();y++) {
				int pix = bi.getRGB(x, y);
				  
				int a = (pix >> 24) & 0xff;
				int r = (pix >> 16) & 0xff;
				int g = (pix >> 8) & 0xff;
				int b = (pix ) & 0xff;
				returnVal[x][y] = (r != 0) || (g != 0) || (b != 0);
			}
		}
		return returnVal;
	}
	
	public  static Vector3f[][] getVecArrayForImageFile(String fileName)  {
		BufferedImage bi;
		try {
			bi = ImageIO.read(new File(fileName));
		} catch (IOException e) {
			e.printStackTrace();
			return null;
		}
		Vector3f[][] returnVal = new Vector3f[bi.getWidth()][bi.getHeight()];
		for(int x = 0; x < bi.getWidth();x++) {
			for(int y = 0; y < bi.getHeight();y++) {
				int pix = bi.getRGB(x, y);
				int a = (pix >> 24) & 0xff;
				int r = (pix >> 16) & 0xff;
				int g = (pix >> 8) & 0xff;
				int b = (pix ) & 0xff;
				returnVal[x][y] = new Vector3f(r-128,g-128,b-128);
			}
		}
		return returnVal;
	}
	public  static void mergVecArrayForImageFile(String fileName, Vector3f[][] vecArr, float srcScale, float destScale)  {
		BufferedImage bi = null;
		try {
			bi = ImageIO.read(new File(fileName));
		} catch (IOException e) {
			e.printStackTrace();
		}
		for(int x = 0; x < bi.getWidth();x++) {
			for(int y = 0; y < bi.getHeight();y++) {
				int pix = bi.getRGB(x, y);
				int a = (pix >> 24) & 0xff;
				int r = (pix >> 16) & 0xff;
				int g = (pix >> 8) & 0xff;
				int b = (pix ) & 0xff;
				vecArr[x][y].scale(srcScale);
				Vector3f v = new Vector3f(r-128,g-128,b-128);
				v.scale(destScale);
				vecArr[x][y].add(v);
			}
		}
	}
}
