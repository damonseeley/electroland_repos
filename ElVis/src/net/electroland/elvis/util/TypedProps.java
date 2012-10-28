package net.electroland.elvis.util;

import java.awt.Point;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Properties;
import java.util.Vector;

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

	public boolean setProperty(String key, boolean defaultValue) {
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
	
	public int inc(String key, int v, int defv) {
		 setProperty(key, getProperty(key, defv)+ v);
		return getProperty(key,defv);
	}

	public double inc(String key, double v, double defv) {
		setProperty(key, getProperty(key, defv)+ v);
		return getProperty(key,defv);
	}

	public float inc(String key, float v, float defv) {
		 setProperty(key, getProperty(key, defv)+ v);
		return getProperty(key,defv);
	}

	public long inc(String key, long v, long defv) {
		 setProperty(key, getProperty(key, defv)+ v);
		return getProperty(key,defv);
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

		 try{
		 	 InputStream is = this.getClass().getClassLoader().getResourceAsStream(path);
			 load(is);
		 }catch(NullPointerException e){
			 System.out.println("Make sure that '" + path + "' is in your classpath.");
			 e.printStackTrace(System.err);
		 }catch(IOException e){
			 System.out.println("Make sure that '" + path + "' is in your classpath.");
			 e.printStackTrace(System.err);
		 }
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
	 // added to alphabetize when store is called
	 @SuppressWarnings("unchecked")
	  public synchronized Enumeration keys() {
	     Enumeration keysEnum = super.keys();
	     Vector keyList = new Vector();
	     while(keysEnum.hasMoreElements()){
	       keyList.add(keysEnum.nextElement());
	     }
	     Collections.sort(keyList);
	     return keyList.elements();
	  }
	 
}
