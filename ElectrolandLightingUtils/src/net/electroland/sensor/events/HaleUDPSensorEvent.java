package net.electroland.sensor.events;

import net.electroland.sensor.SensorEvent;
import net.electroland.util.Util;

public class HaleUDPSensorEvent extends SensorEvent {

	private byte cmdByte;
	private byte[] originalPacket;
	private byte[] data;
	private boolean isValid = false;
	private String sender;

	public HaleUDPSensorEvent(String sender, byte[] packet)
	{
		// who's it from?
		this.sender = sender;

		// is the originalPacket length going to include dead buffer?
		// if so, need to find the location of the stop byte, and ONLY copy
		// that much here.  until we test, assuming the stop byte is the last
		// byte.
		originalPacket = new byte[packet.length];
		System.arraycopy(packet, 0, this.originalPacket, 0, originalPacket.length);

		// verify we got valid Hale
		isValid = originalPacket[0] == (byte)255 &&
					originalPacket[originalPacket.length - 1] == (byte)254;

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