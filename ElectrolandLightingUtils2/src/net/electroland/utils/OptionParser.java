package net.electroland.utils;

import java.io.IOException;
import java.io.InputStream;
import java.util.Enumeration;
import java.util.HashMap;
import java.util.Hashtable;
import java.util.Map;
import java.util.Properties;
import java.util.Set;

/**
 * To do: document.  add validation handling.
 * @author bradley
 *
 */

public class OptionParser {

	private static String ARG_MARKER = " -";

	// main is just a unit test
	public static void main(String args[]){
		Properties p = new Properties();
		p.put("cat.1", "-class goober -width 700");
		p.put("cat.name with pace", "-height 600");
		p.put("dog.mydog", "-weight 600");
		p.put("dog.mydog.2", "-weight 400 -foo phi");
		p.put("dog.fixture[0]", "-weight 400");

		try {
			OptionParser op = new OptionParser(p);

			System.out.println(op.getObjectnames("cat"));
			System.out.println(op.getObjectnames("dog"));
			
			System.out.println(op.getParams("cat", "1"));
			System.out.println(op.getParams("dog", "mydog"));
		
			System.out.println(op.getParam("dog", "mydog.2", "-fooe"));
			System.out.println(op.getRequiredParam("dog", "mydog.2", "-fooe"));
			
		} catch (OptionException e) {
			e.printStackTrace();
		}
		
	}
	
	//  objectType -> named_object -> parameters (key value pairs)
	private Map<String,Map<String,Map<String,String>>> objects;

	/**
	 * Take properties file provided by client
	 * @param p
	 */
	public OptionParser(Properties p) throws OptionException
	{
		this.objects = init(p);
	}

	/**
	 * load resourceName from classpath
	 * @param resourceName
	 */
	public OptionParser(String resourceName) throws OptionException
	{
		Properties p = new Properties();

		InputStream is = new Util().getClass().getClassLoader().getResourceAsStream(resourceName);
		if (is != null)
		{
			try {
				p.load(is);
			} catch (IOException e) {
				throw new OptionException("Please make sure " + resourceName + " is in your classpath.");
			}
		}else{
			throw new OptionException("Please make sure " + resourceName + " is in your classpath.");
		}
		this.objects = init(p);
	}

	public static Map<String,Map<String,Map<String,String>>> init(Properties p) throws OptionException
	{
		// java generics syntax is bullshit and a half!
		Hashtable<String,Map<String,Map<String,String>>> objects 
				= new Hashtable<String,Map<String,Map<String,String>>>();
		
		// get every named properties from the props file
		Enumeration <Object> g = p.keys();
		while (g.hasMoreElements())
		{
			// parse out the namespace from "objectType.name"
			String key = ("" + g.nextElement()).trim();
			
			// make sure there is only one '.'?
			int endOfName = key.indexOf('.');
			if (endOfName == -1 || endOfName == key.length() - 1){
				throw new OptionException("object '" + key + "' requires a name.");
			}

			String objectType = key.substring(0,endOfName);
			System.out.print("objectType: '" + objectType);
			String objectName = key.substring(endOfName + 1, key.length());
			System.out.print("', objectName: '" + objectName);
			Map<String,String> params = parse("" + p.get(key));
			System.out.println("', params: " + params);
			
			// see if the objectType exists or not:
			Map<String,Map<String,String>> names = objects.get(objectType);
			if (names == null)
			{
				names = new Hashtable<String,Map<String,String>>();
				objects.put(objectType, names);
			}

			names.put(objectName, params);
		}

		return objects;
	}
	
	/**
	 * Ask for a type of object, and get the names of all instances of that
	 * object.
	 * 
	 * @param objectType
	 * @return
	 * @throws OptionException
	 */
	public Set<String> getObjectNames(String objectType) throws OptionException
	{
		Map<String,Map<String,String>> type = objects.get(objectType);
		if (type == null){
			throw new OptionException("no object of type '" + objectType + "' was found.");
		}else{
			return type.keySet();
		}
	}
	
	/**
	 * Ask for an object by name of a particular type, and receive all parameters
	 * defined for that object.
	 * 
	 * @param objectType
	 * @param objectName
	 * @return
	 * @throws OptionException
	 */
	public Map<String,String> getParams(String objectType, String objectName) throws OptionException
	{
		Map<String,Map<String,String>> type = objects.get(objectType);
		if (type == null){
			throw new OptionException("no object of type '" + objectType + "' was found.");
		}else{
			Map<String,String> params = type.get(objectName);
			if (params == null)
			{
				throw new OptionException("no object named '" + objectName + "' of type '" + objectType + "' was found.");
			}else
			{
				return params;
			}
		}		
	}

	/**
	 * Ask for a parameter belonging to an object of a particular type and
	 * get it's current value.
	 * 
	 * @param objectType
	 * @param objectName
	 * @param paramName
	 * @return
	 * @throws OptionException
	 */
	public String getParam(String objectType, String objectName, String paramName) throws OptionException
	{
		Map<String,String> params = getParams(objectType, objectName);
		if (!paramName.startsWith("-")){
			paramName = "-" + paramName;
		}
		return params.get(paramName);
	}

	public String getRequiredParam(String objectType, String objectName, String paramName) throws OptionException
	{
		String param = getParam(objectType, objectName, paramName);
		
		if (param == null)
		{			
			throw new OptionException("no parameter '" + paramName + "' in object named '" + objectName + "' of type '" + objectType + "' was found.");
		}else
		{
			return param;
		}
		
	}

	// TODO: make this work.
	public void save()
	{
	
	}

	// TODO: make this work.
	public void setParam(String objectType, String objectName, String paramName, String paramValue)
	{
	
	}

	// TODO: make this work.
	public void setParams(String objectType, String objectName, Map<String,String> params)
	{
		
	}
	
	// TODO: make this work.
	public void addObjectType(String objectType)
	{
		
	}

	// TODO: make this work.
	public void addNamedObject(String objectType, String objectName)
	{
		
	}
	
	/**
	 * This is a primitive parse for [-key val] style option variables in Strings.
	 * For example, for the input:
	 * 
	 * 	-key1 val1 -key2 val2 val3 -key3 val4
	 * 
	 * We'll return a Map with the following key value pairs:
	 * 
	 * 	key		value
	 * 	------	---------
	 * 	-key1	val1
	 * 	-key2	val2 val3
	 * 	-key3	val4
	 * 
	 * WARNING: This parser knowingly converts all tabs to spaces, so any 
	 * 			arguments you pass it that contain tabs will be affected as such.
	 * 
	 * @param str
	 * @return a Map of the keys and their values.
	 * @throws OptionException if the string does not properly start with a flag. 
	 */
	public static Map<String, String> parse(String str) throws OptionException
	{
		HashMap <String, String> map = new HashMap<String, String>();
		if (str == null)
		{
			return map;
		}

		// all we are really doing is tokenizing on " -" (ARG_MARKER).  Then
		// splitting each token on the first space into a kehy and a value.

		// sorry.  no tabs.
		str.replace('\t', ' ');
		// make sure the first and last tokens are delimited.
		str = ' ' + str.trim() + ARG_MARKER; 

		int flagStart = str.indexOf(ARG_MARKER);

		// SPECIAL CASE: first token (or only token) doesn't start with "-".
		if (!str.startsWith(ARG_MARKER))
		{
			throw new OptionException("Unknown option " + 
										str.substring(1, flagStart));
		}

		int realEnd = str.length() - 2;
		while (true)
		{
			if (flagStart == realEnd) // out of tokens.
			{
				break;
			} else {
				int nextFlagStart = str.indexOf(ARG_MARKER, flagStart + 2);
				String tok = str.substring(flagStart, nextFlagStart).trim();
				int flagEnd = tok.indexOf(' ');
				if (flagEnd == -1){
					map.put(tok, null); // SPECIAL CASE: flag has no value.
				}else{
					map.put(tok.substring(0, flagEnd), 
							tok.substring(flagEnd + 1, tok.length()));
					// at some point, we should parse the values attributed to
					// each flag into an object here.
				}
				flagStart = nextFlagStart;
			}
		}
		return map;
	}
}