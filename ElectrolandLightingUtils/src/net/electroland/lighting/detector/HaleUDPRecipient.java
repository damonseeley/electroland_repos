package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;

import net.electroland.util.Util;

import org.apache.log4j.Logger;

public class HaleUDPRecipient extends Recipient {

	public byte getOnVal() {
		return (byte)255;
	}

	public byte getOffVal() {
		return (byte)0;
	}	

	private static Logger logger = Logger.getLogger(HaleUDPRecipient.class);
	private static DatagramSocket socket;

	// for regularly interspersed optional command bytes
	private int interPeriod;	// number of frames between interspersal
	private Byte interCmdByte;	// the optional byte
	private int count = 0;		// our internal count

	public HaleUDPRecipient(String id,
			String ipStr, int port, int channels, Dimension preferredDimensions, 
			Byte interCmdByte, Integer interPeriod)
			throws UnknownHostException {
		super(id, ipStr, port, channels, preferredDimensions);
		this.interCmdByte = interCmdByte;
		this.interPeriod = interPeriod == null ? 1 : interPeriod.intValue();
	}

	public HaleUDPRecipient(String id,
			String ipStr, int port, int channels, Dimension preferredDimensions, 
			String patchgroup, Byte interCmdByte, Integer interPeriod)
			throws UnknownHostException {
		super(id, ipStr, port, channels, preferredDimensions, patchgroup);
		this.interCmdByte = interCmdByte;
		this.interPeriod = interPeriod == null ? 1 : interPeriod.intValue();
	}

	void send(byte[] data) {

		byte[] protocolAndData = new byte[data.length + 3];
		for (int i = 0; i < data.length; i++)
		{
			if (data[i] == (byte)255 ||
				data[i] == (byte)254){
				data[i] = (byte)253;
			}
		}

		System.arraycopy(data, 0, protocolAndData, 2, data.length);

		// start byte
		protocolAndData[0] = (byte)255;

		// command byte
		if (interCmdByte != null && count++ % interPeriod == 0)
		{
			// if an interspersed command byte was specified (and it's time)...
			protocolAndData[1] = interCmdByte.byteValue();
		}else
		{
			protocolAndData[1] = (byte)0; // default command byte
		}

		protocolAndData[protocolAndData.length - 1] = (byte)254;

			logger.debug(this.id + " at IP " + 
								this.ipStr + ":" + 
								Util.bytesToHex(protocolAndData, protocolAndData.length));

		synchronized (this)
		{
			if (socket == null || socket.isClosed())
			{
				try 
				{
					socket = new DatagramSocket(port);
				} catch (SocketException e) {
					logger.error(e);
				}
			}
			DatagramPacket packet = new DatagramPacket(protocolAndData, 
					protocolAndData.length, ip, port);
			try {
				socket.send(packet);
			} catch (IOException e) {
				logger.error(e);
			}
		}
	}
}