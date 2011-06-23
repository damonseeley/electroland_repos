package net.electroland.utils.lighting.protocols;

import java.util.Map;

import net.electroland.utils.OptionException;
import net.electroland.utils.lighting.Recipient;


public class ARTNetRecipient extends Recipient {

	public static int ART_NET_PORT = 6454; // port should be fixed for art net.
	protected int channels, channelBits = 8, universe;
	protected String address;
	
	@Override
	public void configure(Map<String, String> properties)
			throws OptionException {

		// Typical: -channels 512 channelBits 16 -address 127.0.0.1 -universe 1

		// get channels (must be 1-512)
		try{
			Integer channels = Integer.parseInt(properties.get("-channels"));
			if (channels.intValue() < 1 || channels.intValue() > 512)
			{
				throw new OptionException("recipient.channel must be between 1 and 512.");			
			}
			this.channels = channels.intValue();
		}catch(NumberFormatException e)
		{
			throw new OptionException("bad channel value. " + e.getMessage());
		}
		
		// get channelBits - (optional) Valid values are either 8 or 16.
		String channelBitsStr = properties.get("-channelBits");
		if (channelBitsStr != null){
			try{
				Integer channelBits = Integer.parseInt(channelBitsStr);
				if (channelBits.intValue() != 8 && channelBits.intValue() != 16)
				{
					throw new OptionException("recipient.channelBits must be either 8 or 16.");			
				}
				this.channelBits = channelBits.intValue();
			}catch(NumberFormatException e)
			{
				throw new OptionException("bad channelBits value. " + e.getMessage());
			}			
		}
		
		// get IP address (not validated here)
		address = properties.get("-address");
		if (address == null)
		{
			throw new OptionException("recipient.address must be defined.");			
		}
		
		// get universe (must be 0-255)
		try{
			Integer universe = Integer.parseInt(properties.get("-universe"));
			if (universe.intValue() < 0 || universe.intValue() > 255)
			{
				throw new OptionException("recipient.universe must be between 0 and 255.");			
			}
			this.universe = universe.intValue();
		}catch(NumberFormatException e)
		{
			throw new OptionException("bad channel value. " + e.getMessage());
		}
	}

	@Override
	public void allOn() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void allOff() {
		// TODO Auto-generated method stub
		
	}

	@Override
	public void send(byte[] b) {
		// TODO Auto-generated method stub
		
	}

}
