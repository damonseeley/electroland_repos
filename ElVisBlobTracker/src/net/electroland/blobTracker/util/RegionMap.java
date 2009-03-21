package net.electroland.blobTracker.util;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.HashMap;

import javax.imageio.ImageIO;
import net.electroland.blobDetection.Region;

public class RegionMap {
	public Region[] region;
	int[][] map;
	HashMap<Integer, Integer> colorToRegion = new HashMap<Integer, Integer>();
	int regions = 0;

	public RegionMap(String mapFileName) {
		BufferedImage bi;
		try {
			bi = ImageIO.read(new File(mapFileName));
			map  = new int[bi.getWidth()][bi.getHeight()];
			for(int x = 0; x < bi.getWidth(); x++) {
				for(int y = 0; y < bi.getHeight(); y++) {
					int color = bi.getRGB(x, y);
					Integer region = colorToRegion.get(color);
					if(region == null) {
						region = regions++;
						colorToRegion.put(color, region);
						System.out.println("new map " + color + " " + region);
					}
					map[x][y]= region;
				}
			}
		} catch (IOException e) {
			System.out.println("Unable to open region map" + mapFileName);
			regions = 1;
		}
		region = new Region[regions];
		for(int i = 0; i < regions;i++) {
			region[i] = new Region(i);
		}
		
	}
	
	public int size() {
		return regions;
	}
	
	public Region getRegion(int x, int y) {
		if(regions == 1) return region[0];
		return region[map[x][y]];
	}
	public Region getRegion(int i) {
		return region[i];
	}
	

}
