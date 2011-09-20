package net.electroland.utils.sensors;

import java.net.InetAddress;

import net.electroland.utils.ElectrolandProperties;
import net.electroland.utils.OptionException;

public class IOManager {

	private IOThread inputThread;
	
	// for diagnostic purposes
	public static void main(String[] args)
	{
		
	}
	
	public void init() throws OptionException
	{
		init("io.properties");
	}
	
	public void init(String propsFileName) throws OptionException
	{
		ElectrolandProperties op = new ElectrolandProperties(propsFileName);
		
		// set global params
		
		// get all ionodeType objects from the props
		// for each type
		//	find the type's factory class and store it (mapped to type)
		//  call prototypeDevice(op.getObjects(NAME_OF_TYPE))
		// get all ionode objects
		//  find the factory for the type (as appropriate)
		//  call createInstance(REST_OF_INODE_PARAMS)
		//  store the Device, hashed against it's name
		// get all istates objects
		//  for each istate
		//   store id, x,y,z,units and any filters
		//   parse the tag list
		//   for each tag
		//     see if the tag and associated array already exists
		//       no? add the tag to the tag list and map a new array to it
		//     store this istate againt that tag
		//     find the ionode
		//     call patch(state, port)
		//  for each ostate
		//   same exact thing as istates
	}

	public void start()
	{
		
	}

	public void stop()
	{
		
	}

	public IOState getState(String id)
	{
		return null;
	}

	public IOState[] getStates(String tag)
	{
		return null;
	}

	public IOState[] getStatesForDevice(String deviceName)
	{
		return null;
	}
	
	public IOState[] getStatesForIP(InetAddress ip)
	{
		return null;
	}
}