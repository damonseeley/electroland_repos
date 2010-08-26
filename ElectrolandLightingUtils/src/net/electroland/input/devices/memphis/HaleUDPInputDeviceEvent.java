package net.electroland.input.devices.memphis;

import net.electroland.input.InputDeviceEvent;
import net.electroland.util.Util;

public class HaleUDPInputDeviceEvent extends InputDeviceEvent {

	private byte cmdByte;
	private byte[] originalPacket;
	private byte[] data;
	private boolean isValid = false;
	private String sender;

	public HaleUDPInputDeviceEvent(String sender, byte[] packet)
	{
		// who's it from?
		this.sender = sender;

		// find the end byte (well before the end of the packet length, hopefully)
		boolean endFound = false;
		int length = 0;
		while (!endFound && length < packet.length)
		{
			endFound = packet[length++] == (byte)254;
		}

		// make sure there was a start byte
		isValid = endFound && packet[0] == (byte)255;

		// cache the packet
		originalPacket = new byte[length];
		System.arraycopy(packet, 0, this.originalPacket, 0, originalPacket.length);

		if (isValid){
			// yes?  get the command byte
			cmdByte = originalPacket[1];
			
			// and data.
			data = new byte[originalPacket.length - 3];
			System.arraycopy(originalPacket, 2, data, 0, data.length);
		}
	}

	public String toString()
	{
		StringBuffer sb = new StringBuffer("HaleUDPSensorEvent[");
		sb.append("sender=").append(sender);
		sb.append(", isValid=").append(isValid);
		sb.append(", cmdByte=").append(Util.bytesToHex(cmdByte));
		sb.append(", originalPacket=");
		sb.append(Util.bytesToHex(originalPacket, originalPacket.length));
		sb.append(", data=");
		sb.append(Util.bytesToHex(data, data.length));
		
		return sb.append(']').toString();
	}
	
	public byte getCmdByte() {
		return cmdByte;
	}

	public byte[] getOriginalPacket() {
		return originalPacket;
	}

	public byte[] getData() {
		return data;
	}

	public boolean isValid() {
		return isValid;
	}

	public String getSender() {
		return sender;
	}
}