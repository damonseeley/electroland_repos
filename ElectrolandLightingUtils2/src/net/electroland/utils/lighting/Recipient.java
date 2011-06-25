package net.electroland.utils.lighting;

import java.util.Map;

import net.electroland.utils.OptionException;


abstract public class Recipient {
	
	private String name;	
	private CanvasDetector[] channels;
	
	public void setName(String name){
		this.name = name;
	}
	
	public String getName(){
		return name;
	}

	protected void sync(){
		
		byte[] b = new byte[channels.length];
		
		for (int i = 0; i < channels.length; i++)
		{
			//  for each non-null channel, evaluate the CanvasDetector and store
			//  the result in an array of bytes.
		}
		
		send(b);
	}

	
	// configure
	abstract public void configure(Map<String,String> properties) throws OptionException;

	// configure
	abstract public void map(int channel, CanvasDetector cd) throws OptionException;
	
	// send all "on" values
	abstract public void allOn();
	
	// send all "off" values
	abstract public void allOff();

	// protocol specific handling
	abstract public void send(byte[] b);
}