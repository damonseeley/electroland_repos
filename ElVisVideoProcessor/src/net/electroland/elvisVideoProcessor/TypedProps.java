package net.electroland.elvisVideoProcessor;

import java.awt.Point;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.Properties;

public class TypedProps extends Properties {
	

	
	public int setProperty(String key, int value) {
		Object o = setProperty(key, Integer.toString(value));
		if(o == null) {
			return 0;
		} else {
			return Integer.valueOf((String) o) ;
		}
	}

	public float setProperty(String key, float value) {
		Object o = setProperty(key, Float.toString(value));
		if(o == null) {
			return 0;
		} else {
			return Float.valueOf((String) o) ;
		}
	}

	public long setProperty(String key, long value) {
		Object o = setProperty(key, Long.toString(value));
		if(o == null) {
			return 0;
		} else {
			return Long.valueOf((String) o) ;
		}
	}

	public double setProperty(String key, double value) {
		Object o = setProperty(key, Double.toString(value));
		if(o == null) {
			return 0;
		} else {
			return Double.valueOf((String) o) ;
		}
	}
	
	public Point setProperty(String key, Point value) {
		Object o = setProperty(key, pointToString(value));
		if(o == null) {
			return new Point();
		} else {
			return parsePoint((String) o) ;
		}
	}

	private boolean setProperty(String key, boolean defaultValue) {
		Object o = setProperty(key, Boolean.toString(defaultValue));
		if(o == null) {
			return defaultValue;
		} else {
			return Boolean.valueOf((String)o);
		}
		
	}

	/*
	private Bounds setProperty(String key, Bounds defaultValue) {
		Object o = setProperty(key, defaultValue.toString());
		if(o == null) {
			return defaultValue;
		} else {
			return Bounds.valueOf((String)o);
		}
		
	}
	*/
	public boolean getProperty(String key, boolean defaultValue) {
		 String val = getProperty(key);
		 if(val == null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
			 
		 return Boolean.valueOf(val);
		
	}

	public int getProperty(String key, int defaultValue) { 
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return Integer.valueOf(val);		 
	 }
	 
	public float getProperty(String key, float defaultValue) { 
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return Float.valueOf(val);	 
	 }
	 
	public double getProperty(String key, double defaultValue) { 
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return Double.valueOf(val);		 
	 }
	 
	public long getProperty(String key, long defaultValue) { 
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return Long.valueOf(val);	 
	 }

	/*
	public Bounds getProperty(String key, Bounds defaultValue) {
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return Bounds.valueOf(val);	 
		
	}
*/
	public String getProperty(String key, String defaultValue) {
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return val;
		
	}

	public Point getProperty(String key, Point defaultValue) {
		 String val = getProperty(key);
		 if(val==null) {
			 setProperty(key, defaultValue);
			 return defaultValue;
		 }
		 return parsePoint(val);
		
	}
	 public void load(String path) throws IOException {
			load(new FileInputStream(path));
	 }
	 
	 public void store(String path, String comments) throws IOException {
		 store(new FileOutputStream(path), comments);		 
	 }
	 
	 public static String pointToString(Point p) {
		return p.x + "," + p.y;
	 }
	 public static Point parsePoint(String s) {
		 Point p = new Point();
		 String[] vals = s.split(",");
		 p.x = Integer.parseInt(vals[0]);
		 p.y = Integer.parseInt(vals[1]);
		 return p;
	 }
	 
}
