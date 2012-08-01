package net.electroland.presenceGrid.util;

import java.io.IOException;


public class GridProps extends TypedProps {
	String filename;
	
	static GridProps THE_GRID_PROPS = null;
	
	public static GridProps init(String filename) throws IOException {
		THE_GRID_PROPS = new GridProps();
		THE_GRID_PROPS.load(filename);
		return THE_GRID_PROPS;
	}
	

	public static GridProps getGridProps() {
				return THE_GRID_PROPS;
	}
	
	public void store() {
		store(filename);
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
		filename = s;
		try {
			super.load(s);
			System.out.println("loaded " + s);
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
