package net.electroland.gotham.processing.assets;

import java.util.List;

import net.electroland.utils.ElectrolandProperties;
import processing.core.PApplet;
import processing.core.PImage;


public class ColorPalette {
	private static ElectrolandProperties props = new ElectrolandProperties("Gotham-global.properties");;
	PApplet p;
	
	private static int[] colors;
	private static int numSwatches;
	private static int numColors;
	private static List<String> fileNames;
	
	public ColorPalette(PApplet p) {
		this.p = p;
	}

	public void createNewPalette(int n) {
		fileNames = props.getOptionalList("wall", "East", "colorPalettes");
		numSwatches = fileNames.size();
		
		int num = 0;
		String s = fileNames.get(n);
		PImage thisImage = p.loadImage("/../Gotham/depends/images/"+s);
		numColors = (int)(thisImage.width/20);
		colors = new int[numColors]; //Divide swatch with by 20 to get how many colors are int he file. Each square is 20x20 pixels
		for (int i = 10; i < thisImage.width; i += 20) {
			colors[num] = thisImage.get(i, thisImage.height / 2);
			num++;
		}
	}
	public static String getSwatchName(int i){
		return fileNames.get(i);
	}
	
	public static int getRandomColor(){
		return colors[(int) (Math.random() * colors.length)];
	}
	
	//Returns the number of colors in the current swatch
	public static int getNumColors(){
		return numColors;
	}
	
	//Returns the number of jpgs in the props file
	public static int getNumSwatches(){
		return numSwatches;
	}
}
