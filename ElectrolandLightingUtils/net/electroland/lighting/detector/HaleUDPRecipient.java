package net.electroland.lighting.detector;

import java.awt.Dimension;
import java.io.IOException;
import java.net.DatagramPacket;
import java.net.DatagramSocket;
import java.net.SocketException;
import java.net.UnknownHostException;

import org.apache.log4j.Logger;

public class HaleUDPRecipient extends Recipient {

	private static Logger logger = Logger.getLogger(HaleUDPRecipient.class);
	private static DatagramSocket socket;

	public HaleUDPRecipient(String id,
			String ipStr, int port, int channels, Dimension preferredDimensions)
			throws UnknownHostException {
		super(id, ipStr, port, channels, preferredDimensions);
	}

	public HaleUDPRecipient(String id,
			String ipStr, int port, int channels, Dimension preferredDimensions, String patchgroup)
			throws UnknownHostException {
		super(id, ipStr, port, channels, preferredDimensions, patchgroup);
	}

	@Override
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
		
		protocolAndData[0] = (byte)255;
		protocolAndData[1] = (byte)0;
		protocolAndData[protocolAndData.length - 1] = (byte)254;

			logger.debug(this.id + " at IP " + 
								this.ipStr + ":" + 
								bytesToHex(protocolAndData, protocolAndData.length));			

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