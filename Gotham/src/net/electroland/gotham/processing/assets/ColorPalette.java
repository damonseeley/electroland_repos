package net.electroland.gotham.processing.assets;

import java.util.List;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;
import processing.core.PImage;
import processing.core.PApplet;


public class ColorPalette {
	private static ElectrolandProperties props = GothamConductor.props;
	PApplet p;
	private int[] colors;
	private static int numSwatches;
	
	public ColorPalette(PApplet p) {
		this.p = p;
	}

	public int[] getPalette(int n) {
		List<String> fileNames = props.getOptionalList("wall", "East", "colorPalettes");
		numSwatches = fileNames.size();
		colors = new int[15]; //A swatch has 15 colors
		int num = 0;
		String s = fileNames.get(n);
		PImage thisImage = p.loadImage("/../Gotham/depends/images/"+s);

		for (int i = 10; i < thisImage.width; i += 20) {
			colors[num] = thisImage.get(i, thisImage.height / 2);
			num++;
		}
		
		return colors;
	}
	public static int getNumSwatches(){
		return numSwatches;
	}
}
