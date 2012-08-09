package net.electroland.elvis.util;

import java.io.IOException;


public class ElProps extends TypedProps {
	
	/**
	 * 
	 */
	private static final long serialVersionUID = 1L;
	
	private static ElProps THE_PROPS = null;

	public String fileName;
	


	public static ElProps init(String s) {
		ElProps fp = new ElProps();
		fp.load(s);
		return THE_PROPS = fp;
		
	}

	public void store() {
		store(fileName);
	}
	public void store(String s) {
		try {
			super.store(s, "");
			System.out.println("saved " + s);
		} catch (IOException e) {
			e.printStackTrace();
		}		
	}
	
	public void load(String s) {
		fileName = s;
		try {
			System.out.println("loading " + s);
			super.load(s);
		} catch (IOException e) {
			System.out.println(s + " not found will use defaults defaults");
			try {
				store(s, "default values");
			} catch (IOException e1) {
				System.out.println("I/O exception while saving default property file\n" + e1);
			}
		}
	}
	


	

	

}
