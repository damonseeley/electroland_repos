package net.electroland.gotham.processing.assets;

import java.util.List;
import net.electroland.gotham.core.GothamConductor;
import net.electroland.utils.ElectrolandProperties;
import net.electroland.gotham.processing.EastBlurTest;
import processing.core.PImage;
import processing.core.PApplet;


public class ColorPalette {
	private static ElectrolandProperties props = GothamConductor.props;
	private int[] colors;
	
	public ColorPalette(PApplet p) {
		List<String> fileNames = props.getOptionalList("wall", "East", "colorPalettes");
		colors = new int[15]; //A swatch has 15 colors
		int num = 0;
		String s = fileNames.get(EastBlurTest.selector);
		System.out.println(s);
		PImage thisImage = p.loadImage("/../Gotham/depends/images/"+s);

		for (int i = 10; i < thisImage.width; i += 20) {
			colors[num] = thisImage.get(i, thisImage.height / 2);
			num++;
		}
	}

	public int[] getPalette() {
		return colors;
	}
}
