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
		
		Byte[] b = new Byte[channels.length];
		
		for (int i = 0; i < channels.length; i++)
		{
			if (channels[i] != null){
				b[i] = channels[i].latestState;
			}
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
	abstract public void send(Byte[] b);

	/**
	 * print debugging info.
	 */
	abstract public void debug();

}