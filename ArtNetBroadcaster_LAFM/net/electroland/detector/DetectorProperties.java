package net.electroland.detector;

import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.Properties;

public class DetectorProperties extends Properties {

	public final String FIXTURE = "fixture";
	public final String DETECTOR = "detector";
	public final String LIGHT = "light";
	public final String GROUP = "group";
	
	@Override
	public synchronized void load(InputStream inStream) throws IOException {
		super.load(inStream);
		
		Enumeration <?> names = this.propertyNames();
		while (names.hasMoreElements()){
			String name = (String)names.nextElement();
			this.put(name, parse(name, this.getProperty(name)));
		}
		
		//
		// parse every line.  Either hash the id to the appropriate object,
		// or store the parsing exception.
		//
		// how to create relationships?
	}
	
	public Object parse(String name, String str){
		return null;
	}
}