package net.electroland.lafm.core;

import java.text.DecimalFormat;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Properties;
import java.util.StringTokenizer;

import org.apache.log4j.Logger;

import processing.core.PApplet;
import processing.core.PImage;

public class ImageSequenceCache {

	static Logger logger = Logger.getLogger(ImageSequenceCache.class);
	private HashMap <String, PImage[]> cache;

	public ImageSequenceCache(Properties p, PApplet applet){

		cache = new HashMap <String, PImage[]>();

		// try to parse and add everything in imges.properties.
		Enumeration <?> e = p.propertyNames();
		while (e.hasMoreElements()){
			String name = (String)e.nextElement();
			logger.info("loading image sequence: " + name);
			cache.put(name, parse(p.getProperty(name), cache, applet));
		}
	}

	// this is all anyone should be calling.
	public PImage[] getSequence(String sequenceName){
		return cache.get(sequenceName);
	}
	
	private static PImage[] parse(String property, HashMap <String, PImage[]>cache, PApplet applet){

		StringTokenizer st = new StringTokenizer(property, ",\t");

		String prefix = st.nextToken().trim();	// the root of the filename
		int start = Integer.parseInt(st.nextToken().trim());	// the start file number
		int end = Integer.parseInt(st.nextToken().trim());	// the end file number
		DecimalFormat d = new DecimalFormat(st.nextToken().trim());	// the number format for the file number (see java.text.DecimalFormat)
		String suffix = st.nextToken().trim();	// the end of tail end of the filename.

		// should throw some exceptions here if start and end are invalid, etc.
		
		PImage[] sequence = new PImage[end - start + 1];

		for (int i = start; i <= end; i++){
			logger.info("loading: " + prefix + d.format(i) + suffix);
			sequence[i - start] = applet.loadImage(prefix + d.format(i) + suffix);
		}

		return sequence;
	}
}